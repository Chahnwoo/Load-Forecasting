#!/usr/bin/env python3
"""Acquire small NCSS subsets of NSF NCAR GDEX d084001 for strict backtests."""
from __future__ import annotations
import argparse, json, re, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from src.backtesting.acquisition import (GDEX_DATASET, GDEX_MODEL, GDEX_REFERENCE, GDEX_SOURCE,
    GDEX_ROOT, gdex_ncss_url, gdex_object_url, hourly_from_gdex, parse_gdex_name, sha256_file, write_manifest)
from src.backtesting.strict_dataset import operating_intervals

CANONICAL_VARIABLES = {
 "temperature_2m": "Temperature_height_above_ground",
 "relative_humidity_2m": "Relative_humidity_height_above_ground",
 "u_wind_10m": "u-component_of_wind_height_above_ground",
 "v_wind_10m": "v-component_of_wind_height_above_ground",
}

STATISTICAL_VARIABLES = {
 "precipitation": ("Total precipitation", "Ground or water surface", "Accumulation"),
 "shortwave_radiation": ("Downward Short-Wave Radiation Flux", "Ground or water surface", "Average"),
}

CLOUD_COVER_SEMANTICS = ("Total cloud cover", "Entire atmosphere", "UnknownStatType--1")

TRANSIENT_NCSS_HTTP_STATUSES = {404, 408, 429, 500, 502, 503, 504}
MAX_NCSS_RETRY_DELAY_SECONDS = 8
OPENDAP_METADATA_ATTEMPTS = 5

def required_objects(month):
    names=set()
    for _, day in operating_intervals(month).groupby("operating_day"):
        cutoff=day.forecast_cutoff_utc.iloc[0]; cycle=cutoff.floor("6h")
        while cycle+pd.Timedelta(hours=10)>cutoff: cycle-=pd.Timedelta(hours=6)
        for valid in day.time_utc:
            lead=int((valid-cycle).total_seconds()/3600)
            if not 0 <= lead <= 240: raise RuntimeError(f"d084001 3-hour range cannot cover {valid} from {cycle}")
            # Instantaneous quantities are constructed from the adjacent products
            # belonging to this same eligible cycle.  Exact products need no upper
            # neighbour; set semantics deduplicate brackets shared by target hours.
            lower=lead-(lead % 3); upper=lower if lead == lower else lower+3
            for product_lead in (lower,upper):
                names.add(f"gfs.0p25.{cycle:%Y%m%d%H}.f{product_lead:03d}.grib2")
    return sorted(names)

def _opendap_das_url(name):
    parsed=parse_gdex_name(name); day=parsed["model_init_time_utc"].strftime("%Y%m%d")
    return (f"{GDEX_ROOT}/dodsC/files/g/{GDEX_DATASET}/{day[:4]}/{day}/"
            f"{parsed['filename']}.das")

def _das_variables(content):
    """Read top-level OPeNDAP DAS variable blocks and their scalar attributes."""
    text=content.decode("utf-8") if isinstance(content,bytes) else content
    opening=text.find("{")
    if opening < 0: raise RuntimeError("OPeNDAP DAS has no Attributes block")
    variables=[]; position=opening+1
    block_start=re.compile(r"([^\s{}]+)\s*\{")
    attribute=re.compile(r'\b(?:Byte|Int\d+|UInt\d+|Float\d+|String|Url)\s+([^\s]+)\s+(.+?)\s*;',re.S)
    while match := block_start.search(text,position):
        depth=1; cursor=match.end(); quoted=False; escaped=False
        while cursor < len(text) and depth:
            char=text[cursor]
            if quoted:
                if escaped: escaped=False
                elif char == "\\": escaped=True
                elif char == '"': quoted=False
            elif char == '"': quoted=True
            elif char == "{": depth+=1
            elif char == "}": depth-=1
            cursor+=1
        if depth: raise RuntimeError(f"unterminated OPeNDAP DAS block {match.group(1)!r}")
        body=text[match.end():cursor-1]; attrs={}
        for item in attribute.finditer(body):
            value=item.group(2).strip()
            quoted_values=re.findall(r'"((?:\\.|[^"\\])*)"',value)
            attrs[item.group(1)]=", ".join(v.replace('\\"','"') for v in quoted_values) if quoted_values else value
        variables.append((match.group(1),attrs)); position=cursor
    return variables

def _metadata_value(attrs, *names):
    lowered={key.lower():str(value).strip() for key,value in attrs.items()}
    return next((lowered[name.lower()] for name in names if name.lower() in lowered),"")

def _normal(value):
    return " ".join(re.sub(r"[^a-z0-9]+"," ",value.lower()).split())

def _diagnostic(variables, terms):
    related=[]
    words=set(_normal(" ".join(terms)).split())
    for name,attrs in variables:
        parameter=_metadata_value(attrs,"Grib2_Parameter_Name")
        if words & set(_normal(name+" "+parameter).split()):
            related.append({"name":name,"parameter":parameter,
                "level":_metadata_value(attrs,"Grib2_Level_Desc","Grib2_Level_Type"),
                "statistical_process":_metadata_value(attrs,"Grib2_Statistical_Process_Type","Grib_Statistical_Interval_Type"),
                "units":_metadata_value(attrs,"units")})
    return related[:20]

def _opendap_metadata(session, endpoint, attempts=OPENDAP_METADATA_ATTEMPTS):
    """Retrieve DAS bytes with bounded retries and actionable HTTP errors."""
    for attempt in range(attempts):
        try:
            response=session.get(endpoint,timeout=(20,120))
        except requests.RequestException:
            if attempt+1 == attempts: raise
            time.sleep(min(MAX_NCSS_RETRY_DELAY_SECONDS,2**attempt))
            continue
        status=getattr(response,"status_code",200)
        headers=getattr(response,"headers",{})
        content_type=headers.get("content-type","")
        final_url=getattr(response,"url",None) or endpoint
        content=response.content
        if status != 200:
            body=content.decode("utf-8",errors="replace")[:1000]
            error=RuntimeError(
                f"OPeNDAP metadata retrieval failed: HTTP {status}; "
                f"content-type={content_type!r}; response body={body!r}; "
                f"final request URL={final_url}")
            if status not in TRANSIENT_NCSS_HTTP_STATUSES or attempt+1 == attempts:
                raise error
            retry_after=headers.get("retry-after")
            try: delay=float(retry_after) if retry_after is not None else 2**attempt
            except (TypeError,ValueError): delay=2**attempt
            time.sleep(min(MAX_NCSS_RETRY_DELAY_SECONDS,max(0,delay)))
            continue
        if not content:
            raise RuntimeError(f"OPeNDAP metadata response body is empty; final request URL={final_url}")
        media_type=content_type.partition(";")[0].strip().lower()
        if media_type in {"text/html","application/xhtml+xml"} or re.match(br"\s*<(?:!doctype\s+html|html)\b",content,re.I):
            raise RuntimeError(
                f"OPeNDAP metadata response is HTML; content-type={content_type!r}; "
                f"final request URL={final_url}")
        return content
    raise AssertionError("unreachable OPeNDAP retry state")

def discover_variables(session, name):
    endpoint=_opendap_das_url(name)
    variables=_das_variables(_opendap_metadata(session,endpoint))
    if not variables:
        raise RuntimeError(f"OPeNDAP DAS yielded no metadata blocks; final request URL={endpoint}")
    found={}
    for field, canonical in CANONICAL_VARIABLES.items():
        candidates=[name for name,_ in variables if name == canonical]
        if len(candidates)!=1:
            detail=_diagnostic(variables,canonical.split("_"))
            raise RuntimeError(f"OPeNDAP metadata expected one {field}, found {candidates}; candidates={detail}")
        found[field]=candidates[0]
    parameter,level,process=CLOUD_COVER_SEMANTICS
    candidates=[]
    for variable,attrs in variables:
        actual_parameter=_metadata_value(attrs,"Grib2_Parameter_Name")
        actual_level=_metadata_value(attrs,"Grib2_Level_Desc","Grib2_Level_Type")
        actual_process=_metadata_value(attrs,"Grib2_Statistical_Process_Type")
        if (_normal(actual_parameter)==_normal(parameter) and
                _normal(level) in _normal(actual_level) and
                _normal(actual_process)==_normal(process)):
            candidates.append(variable)
    if len(candidates)!=1:
        detail=_diagnostic(variables,(parameter,level,process))
        raise RuntimeError(f"OPeNDAP metadata expected one cloud_cover, found {candidates}; candidates={detail}")
    found["cloud_cover"]=candidates[0]
    for field,(parameter,level,process) in STATISTICAL_VARIABLES.items():
        candidates=[]
        for variable,attrs in variables:
            actual_parameter=_metadata_value(attrs,"Grib2_Parameter_Name")
            actual_level=_metadata_value(attrs,"Grib2_Level_Desc","Grib2_Level_Type")
            actual_process=" ".join(filter(None,(
                _metadata_value(attrs,"Grib2_Statistical_Process_Type"),
                _metadata_value(attrs,"Grib_Statistical_Interval_Type"))))
            if (_normal(actual_parameter)==_normal(parameter) and
                    _normal(level) in _normal(actual_level) and
                    (process is None or _normal(process) in _normal(actual_process))):
                candidates.append(variable)
        if len(candidates)!=1:
            detail=_diagnostic(variables,(parameter,level,process or ""))
            raise RuntimeError(f"OPeNDAP metadata expected one {field}, found {candidates}; candidates={detail}")
        found[field]=candidates[0]
    return found, endpoint

def download(session,url,path,retries=4):
    sidecar=path.with_suffix(path.suffix+".provenance.json")
    if path.exists() and sidecar.exists():
        meta=json.loads(sidecar.read_text())
        if meta.get("ncss_request_url")!=url or meta.get("checksum")!=sha256_file(path):
            raise RuntimeError(f"existing NCSS subset provenance mismatch: {path}")
        return meta["retrieved_at_utc"]
    if path.exists() or sidecar.exists(): raise RuntimeError(f"incomplete prior NCSS state: {path}")
    for attempt in range(retries):
        try:
            response=session.get(url,timeout=(20,300))
            content_type=response.headers.get("content-type", "")
            final_url=getattr(response,"url",None) or url
            if response.status_code != 200:
                body=response.content.decode("utf-8",errors="replace")[:1000]
                error=RuntimeError(
                    f"NCSS download failed: HTTP {response.status_code}; "
                    f"content-type={content_type!r}; response body={body!r}; "
                    f"final request URL={final_url}")
                if response.status_code not in TRANSIENT_NCSS_HTTP_STATUSES or attempt+1==retries:
                    raise error
                retry_after=response.headers.get("retry-after")
                try: delay=float(retry_after) if retry_after is not None else 2**attempt
                except ValueError: delay=2**attempt
                time.sleep(min(MAX_NCSS_RETRY_DELAY_SECONDS,max(0,delay)))
                continue
            media_type=content_type.partition(";")[0].strip().lower()
            if media_type not in {"application/x-netcdf", "application/netcdf"}:
                raise RuntimeError(
                    f"NCSS response is not NetCDF-compatible: HTTP {response.status_code}; "
                    f"content-type={content_type!r}; final request URL={final_url}")
            if not response.content:
                raise RuntimeError(f"NCSS response body is empty; final request URL={final_url}")
            if not response.content.startswith((b"CDF\x01",b"CDF\x02")):
                raise RuntimeError(
                    f"NCSS response has invalid NetCDF signature; content-type={content_type!r}; "
                    f"final request URL={final_url}")
            tmp=path.with_suffix(path.suffix+".part"); tmp.write_bytes(response.content); tmp.replace(path)
            retrieved=datetime.now(timezone.utc).isoformat()
            sidecar.write_text(json.dumps({"ncss_request_url":url,"retrieved_at_utc":retrieved,
                "checksum":sha256_file(path)},indent=2)+"\n")
            return retrieved
        except requests.RequestException:
            if attempt+1==retries: raise
            time.sleep(min(MAX_NCSS_RETRY_DELAY_SECONDS,2**attempt))

def statistical_time_bounds(ds, data_array, parsed, field):
    """Return authoritative CF interval leads for one NCSS backing object."""
    time_coordinates=[coord for coord in data_array.coords.values() if coord.name in data_array.dims and (
        coord.name.lower().startswith("time") or
        str(coord.attrs.get("standard_name", "")).lower() == "time" or
        str(coord.attrs.get("axis", "")).upper() == "T")]
    if len(time_coordinates) != 1:
        raise RuntimeError(f"{field} expected one associated time coordinate; found {[c.name for c in time_coordinates]}")
    candidates=[coord for coord in time_coordinates if coord.attrs.get("bounds")]
    if len(candidates) != 1:
        raise RuntimeError(f"{field} expected one time coordinate with a bounds attribute; found {[c.name for c in candidates]}")
    coordinate=candidates[0]
    bounds_name=str(coordinate.attrs["bounds"])
    if bounds_name not in ds.variables:
        raise RuntimeError(f"{field} time bounds variable {bounds_name!r} is missing")
    bounds=ds[bounds_name]
    if bounds.shape != (1,2):
        raise RuntimeError(f"{field} time bounds must contain exactly one lower/upper pair; shape is {bounds.shape}")
    raw=bounds.values.reshape(-1)
    init=pd.Timestamp(parsed["model_init_time_utc"])
    if init.tzinfo is None: init=init.tz_localize("UTC")
    else: init=init.tz_convert("UTC")
    if pd.api.types.is_datetime64_any_dtype(raw.dtype):
        leads=[(pd.Timestamp(value).tz_localize("UTC")-init).total_seconds()/3600 for value in raw]
    else:
        units=str(bounds.attrs.get("units") or coordinate.attrs.get("units", "")).lower()
        unit_match=re.match(r"^hours?\s+since\s+(.+)$",units)
        if not unit_match:
            raise RuntimeError(f"{field} time bounds require hour units; got {units!r}")
        try: reference=pd.Timestamp(unit_match.group(1))
        except ValueError as exc: raise RuntimeError(f"{field} has invalid CF time units {units!r}") from exc
        if reference.tzinfo is None: reference=reference.tz_localize("UTC")
        else: reference=reference.tz_convert("UTC")
        leads=[((reference+pd.Timedelta(hours=float(value)))-init).total_seconds()/3600 for value in raw]
    lower,upper=leads; period=upper-lower
    if not period > 0:
        raise RuntimeError(f"{field} statistical interval must be positive; bounds are {leads}")
    expected=float(parsed["forecast_lead_hours"])
    if upper != expected:
        raise RuntimeError(f"{field} interval endpoint {upper:g} does not match backing-object forecast lead {expected:g}")
    prefix="shortwave" if field == "shortwave_radiation" else field
    return {f"{prefix}_statistical_interval_start_lead_hours":lower,
            f"{prefix}_statistical_interval_end_lead_hours":upper,
            f"{prefix}_statistical_interval_hours":period,
            f"{prefix}_statistical_interval_start_utc":init+pd.Timedelta(hours=lower),
            f"{prefix}_statistical_interval_end_utc":init+pd.Timedelta(hours=upper)}

def _select_height(data_array, desired_metres, field):
    """Select the DataArray's own vertical coordinate, never a guessed name."""
    candidates=[]
    for name,coord in data_array.coords.items():
        attrs=" ".join(str(v) for v in coord.attrs.values()).lower()
        units=str(coord.attrs.get("units", "")).lower()
        if name in data_array.dims and units in {"m", "meter", "meters", "metre", "metres"} and (
                "height" in name.lower() or "height" in attrs or "above ground" in attrs):
            candidates.append(name)
    if len(candidates) != 1:
        raise RuntimeError(f"{field} expected one associated height coordinate, found {candidates}")
    coordinate=candidates[0]
    available=[float(value) for value in data_array[coordinate].values.reshape(-1)]
    matches=[index for index,value in enumerate(available) if abs(value-desired_metres) < 1e-6]
    if len(matches) != 1:
        raise RuntimeError(f"{field} requires {desired_metres:g} m; available levels are {available}")
    return data_array.isel({coordinate: matches[0]})

def _validate_units(data_array, field):
    units=str(data_array.attrs.get("units", "")).strip().lower().replace(" ", "")
    allowed={"temperature_2m": {"k", "kelvin"}, "relative_humidity_2m": {"%", "percent", "percentage"},
             "u_wind_10m": {"m/s", "ms-1", "m.s-1", "metersecond-1", "metresecond-1"},
             "v_wind_10m": {"m/s", "ms-1", "m.s-1", "metersecond-1", "metresecond-1"}}
    if field in allowed and units not in allowed[field]:
        raise RuntimeError(f"{field} has incompatible units {data_array.attrs.get('units')!r}")

def extract(path,variables,stations,parsed):
    import xarray as xr
    rows=[]
    # NCSS Grid returns classic/64-bit-offset NetCDF-3, not HDF5-backed NetCDF-4.
    with xr.open_dataset(path,engine="scipy") as ds:
        lat_name=next(n for n in ("lat","latitude") if n in ds.coords)
        lon_name=next(n for n in ("lon","longitude") if n in ds.coords)
        statistical={}
        for field in ("precipitation","shortwave_radiation"):
            statistical.update(statistical_time_bounds(ds,ds[variables[field]],parsed,field))
        for station in stations.itertuples():
            lon=station.longitude % 360 if float(ds[lon_name].max())>180 else station.longitude
            values={}
            for field,var in variables.items():
                array=ds[var]
                if field in {"temperature_2m","relative_humidity_2m","u_wind_10m","v_wind_10m"}:
                    level=2 if field in {"temperature_2m","relative_humidity_2m"} else 10
                    array=_select_height(array,level,field)
                    _validate_units(array,field)
                value=array.sel({lat_name:station.latitude,lon_name:lon},method="nearest").squeeze()
                if value.size != 1: raise RuntimeError(f"{field} did not reduce to one grid value; dimensions are {value.dims}")
                values[field]=float(value.values)
            values["temperature_2m"]-=273.15
            values.update(statistical)
            values["precipitation_period_hours"]=statistical["precipitation_statistical_interval_hours"]
            values["shortwave_period_hours"]=statistical["shortwave_statistical_interval_hours"]
            rows.append({"station":station.station_name,"latitude":station.latitude,"longitude":station.longitude,
                "population_weight":station.population_weight,**parsed,**values})
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--month",default="2025-12")
    p.add_argument("--output-root",default="data/backtesting"); p.add_argument("--stations",default="data/stations_population_weights.csv")
    p.add_argument("--source-document",action="append",required=True); p.add_argument("--smoke-test",action="store_true")
    a=p.parse_args(); month_dir=Path(a.output_root)/a.month; raw=month_dir/"raw/gdex_ncss"; processed=month_dir/"processed"
    raw.mkdir(parents=True,exist_ok=True); processed.mkdir(parents=True,exist_ok=True)
    stations=pd.read_csv(a.stations).query("region == 'caiso'"); objects=required_objects(a.month)
    if a.smoke_test: objects=["gfs.0p25.2025113000.f030.grib2"]
    records=[]; extracted=[]
    with requests.Session() as session:
      pad=.15; bbox=dict(north=stations.latitude.max()+pad,south=stations.latitude.min()-pad,east=stations.longitude.max()+pad,west=stations.longitude.min()-pad)
      for name in objects:
        # Statistical variable identifiers encode their product-specific interval,
        # so discovery is deliberately repeated for every backing object.
        variables,metadata_url=discover_variables(session,name)
        parsed=parse_gdex_name(name); request_url,params=gdex_ncss_url(name,list(variables.values()),**bbox)
        target=raw/(name+".nc"); retrieved=download(session,request_url,target); checksum=sha256_file(target)
        for row in extract(target,variables,stations,parsed):
          row.update(source=GDEX_SOURCE,model=GDEX_MODEL,dataset=GDEX_DATASET,dataset_reference=GDEX_REFERENCE,
            backing_source_object=name,backing_fileserver_url=gdex_object_url(name),ncss_request_url=request_url,
            metadata_url=metadata_url,
            raw_subset_local_filename=str(target),subset_retrieved_at_utc=retrieved,checksum=checksum,
            source_object=request_url,available_at_utc=parsed["model_init_time_utc"]+pd.Timedelta(hours=10),availability_policy="gfs_init_plus_10h_v1")
          extracted.append(row)
        records.append({"source":GDEX_SOURCE,"model":GDEX_MODEL,"dataset":GDEX_DATASET,"dataset_reference":GDEX_REFERENCE,
          "backing_source_object":name,"backing_fileserver_url":gdex_object_url(name),"source_object":request_url,
          "ncss_request_url":request_url,"request_parameters":params,"metadata_url":metadata_url,"discovered_variables":variables,
          "raw_subset_local_filename":str(target),"checksum":checksum,"retrieved_at_utc":retrieved,**parsed,
          "availability_policy":"gfs_init_plus_10h_v1","available_at_utc":parsed["model_init_time_utc"]+pd.Timedelta(hours=10)})
    if a.smoke_test:
      write_manifest(month_dir/"acquisition_manifest.gfs.smoke.json",records,month=a.month,source_documents=a.source_document); return 0
    hourly_parts=[]; frame=pd.DataFrame(extracted)
    for _,cycle in frame.groupby("model_init_time_utc"):
      hourly=hourly_from_gdex(cycle); init=pd.to_datetime(hourly.model_init_time_utc,utc=True)
      hourly["forecast_lead_hours"]=(pd.to_datetime(hourly.valid_time_utc,utc=True)-init).dt.total_seconds()/3600
      hourly["available_at_utc"]=init+pd.Timedelta(hours=10); hourly["availability_policy"]="gfs_init_plus_10h_v1"; hourly["source"]=GDEX_SOURCE; hourly["model"]=GDEX_MODEL
      for station,indices in hourly.groupby("station").groups.items():
       source=cycle[cycle.station==station]; hourly.loc[indices,"population_weight"]=source.population_weight.iloc[0]
       hourly.loc[indices,"source_object"]="|".join(sorted(source.ncss_request_url.unique())); hourly.loc[indices,"checksum"]="|".join(sorted(source.checksum.unique()))
      hourly_parts.append(hourly)
    pd.concat(hourly_parts,ignore_index=True).to_csv(processed/"weather_vintages.csv",index=False)
    write_manifest(month_dir/"acquisition_manifest.gfs.json",records,month=a.month,source_documents=a.source_document)
    return 0
if __name__ == "__main__": raise SystemExit(main())
