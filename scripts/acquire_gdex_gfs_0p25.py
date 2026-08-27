#!/usr/bin/env python3
"""Acquire small NCSS subsets of NSF NCAR GDEX d084001 for strict backtests."""
from __future__ import annotations
import argparse, json, re, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from xml.etree import ElementTree
from src.backtesting.acquisition import (GDEX_DATASET, GDEX_MODEL, GDEX_REFERENCE, GDEX_SOURCE,
    gdex_ncss_url, gdex_object_url, hourly_from_gdex, parse_gdex_name, sha256_file, write_manifest)
from src.backtesting.strict_dataset import operating_intervals

PHYSICAL_FIELDS = {
 "temperature_2m": ("temperature", "2 m"), "relative_humidity_2m": ("relative humidity", "2 m"),
 "cloud_cover": ("total cloud",), "u_wind_10m": ("u-component", "10 m"),
 "v_wind_10m": ("v-component", "10 m"), "precipitation": ("total precipitation",),
 "shortwave_radiation": ("downward short-wave",),
}

def required_objects(month):
    names=set()
    for _, day in operating_intervals(month).groupby("operating_day"):
        cutoff=day.forecast_cutoff_utc.iloc[0]; cycle=cutoff.floor("6h")
        while cycle+pd.Timedelta(hours=10)>cutoff: cycle-=pd.Timedelta(hours=6)
        for valid in day.time_utc:
            lead=int((valid-cycle).total_seconds()/3600)
            if not 0 <= lead <= 120: raise RuntimeError(f"d084001 hourly range cannot cover {valid} from {cycle}")
            names.add(f"gfs.0p25.{cycle:%Y%m%d%H}.f{lead:03d}.grib2")
    return sorted(names)

def discover_variables(session, name):
    url,_=gdex_ncss_url(name,["placeholder"],north=42,south=32,east=-114,west=-125)
    endpoint=url.split("?",1)[0]+"/dataset.xml"
    response=session.get(endpoint,timeout=(20,120)); response.raise_for_status()
    root=ElementTree.fromstring(response.content); variables=[]
    for node in root.iter():
        if node.tag.rsplit("}",1)[-1].lower() == "variable":
            key=node.attrib.get("name") or node.attrib.get("vocabulary_name")
            description=" ".join(node.attrib.values()).lower()
            if key: variables.append((key,description))
    found={}
    for field, needles in PHYSICAL_FIELDS.items():
        candidates=[name for name,desc in variables if all(n in desc for n in needles)]
        if len(candidates)!=1: raise RuntimeError(f"NCSS metadata expected one {field}, found {candidates}")
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
            response=session.get(url,timeout=(20,300)); response.raise_for_status()
            tmp=path.with_suffix(path.suffix+".part"); tmp.write_bytes(response.content); tmp.replace(path)
            retrieved=datetime.now(timezone.utc).isoformat()
            sidecar.write_text(json.dumps({"ncss_request_url":url,"retrieved_at_utc":retrieved,
                "checksum":sha256_file(path)},indent=2)+"\n")
            return retrieved
        except requests.RequestException:
            if attempt+1==retries: raise
            time.sleep(2**attempt)

def interval_hours(attrs, field):
    text=" ".join(str(v) for v in attrs.values())
    matches=re.findall(r"(\d+)\s*[-–]\s*(\d+)\s*hour",text,re.I)
    if not matches: raise RuntimeError(f"{field} lacks a GRIB statistical interval in NCSS metadata: {attrs}")
    start,end=map(int,matches[-1]); period=end-start
    if period<=0: raise RuntimeError(f"invalid {field} interval: {attrs}")
    return period

def extract(path,variables,stations,parsed):
    import xarray as xr
    rows=[]
    with xr.open_dataset(path) as ds:
        lat_name=next(n for n in ("lat","latitude") if n in ds.coords)
        lon_name=next(n for n in ("lon","longitude") if n in ds.coords)
        for station in stations.itertuples():
            lon=station.longitude % 360 if float(ds[lon_name].max())>180 else station.longitude
            values={}
            for field,var in variables.items():
                value=ds[var].sel({lat_name:station.latitude,lon_name:lon},method="nearest").squeeze()
                values[field]=float(value.values)
            values["temperature_2m"]-=273.15
            values["precipitation_period_hours"]=interval_hours(ds[variables["precipitation"]].attrs,"precipitation")
            values["shortwave_period_hours"]=interval_hours(ds[variables["shortwave_radiation"]].attrs,"shortwave")
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
      variables,metadata_url=discover_variables(session,objects[0])
      pad=.15; bbox=dict(north=stations.latitude.max()+pad,south=stations.latitude.min()-pad,east=stations.longitude.max()+pad,west=stations.longitude.min()-pad)
      for name in objects:
        parsed=parse_gdex_name(name); request_url,params=gdex_ncss_url(name,list(variables.values()),**bbox)
        target=raw/(name+".nc"); retrieved=download(session,request_url,target); checksum=sha256_file(target)
        for row in extract(target,variables,stations,parsed):
          row.update(source=GDEX_SOURCE,model=GDEX_MODEL,dataset=GDEX_DATASET,dataset_reference=GDEX_REFERENCE,
            backing_source_object=name,backing_fileserver_url=gdex_object_url(name),ncss_request_url=request_url,
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
