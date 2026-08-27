Attributes {
    Temperature_height_above_ground {
        String Grib2_Parameter_Name "Temperature";
        String Grib2_Level_Desc "Specified height level above ground";
        String units "K";
    }
    Relative_humidity_height_above_ground {
        String Grib2_Parameter_Name "Relative humidity";
        String Grib2_Level_Desc "Specified height level above ground";
        String units "%";
    }
    u-component_of_wind_height_above_ground {
        String Grib2_Parameter_Name "u-component of wind";
        String Grib2_Level_Desc "Specified height level above ground";
        String units "m/s";
    }
    v-component_of_wind_height_above_ground {
        String Grib2_Parameter_Name "v-component of wind";
        String Grib2_Level_Desc "Specified height level above ground";
        String units "m/s";
    }
    Total_cloud_cover_entire_atmosphere {
        String Grib2_Parameter_Name "Total cloud cover";
        String Grib2_Level_Type "Entire atmosphere";
        String Grib2_Level_Desc "Entire atmosphere (considered as a single layer)";
        String Grib2_Statistical_Process_Type "UnknownStatType--1";
        String units "%";
    }
    Total_cloud_cover_entire_atmosphere_6_Hour_Average {
        String Grib2_Parameter_Name "Total cloud cover";
        String Grib2_Level_Type "Entire atmosphere";
        String Grib2_Level_Desc "Entire atmosphere (considered as a single layer)";
        String Grib2_Statistical_Process_Type "Average";
        String Grib_Statistical_Interval_Type "Average";
        String units "%";
    }
    Total_precipitation_surface_Mixed_intervals_Accumulation {
        String Grib2_Parameter_Name "Total precipitation";
        String Grib2_Level_Desc "Ground or water surface";
        String Grib2_Statistical_Process_Type "Accumulation";
        String Grib_Statistical_Interval_Type "Mixed intervals";
        String units "kg.m-2";
    }
    Downward_Short-Wave_Radiation_Flux_surface_3_Hour_Average {
        String Grib2_Parameter_Name "Downward Short-Wave Radiation Flux";
        String Grib2_Level_Desc "Ground or water surface";
        String Grib2_Statistical_Process_Type "Average";
        String Grib_Statistical_Interval_Type "3 hour";
        String units "W.m-2";
    }
    Convective_precipitation_surface_3_Hour_Accumulation {
        String Grib2_Parameter_Name "Convective precipitation";
        String Grib2_Level_Desc "Ground or water surface";
        String Grib2_Statistical_Process_Type "Accumulation";
        String units "kg.m-2";
    }
    NC_GLOBAL {
        String title "GFS 0.25 degree fixture";
    }
}
