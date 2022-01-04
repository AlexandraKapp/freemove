library(sf)
library(dplyr)
library(readr)

# read in csv
df <- read_tsv("tapas2movinglab_2m_offset_100trips.csv")
colnames(df)

# create a geo linestring object from text data
df <- df %>%
  mutate(geometry = paste0("Linestring(", `LEG_SHAPE|geometry`, ")"))
df <- st_as_sf(df, wkt = "geometry") %>%
  select(-"LEG_SHAPE|geometry")
st_crs(df) <- 4326

# view data on map
mapview::mapview(df, zcol = "TRAFFIC_MODE|text")

# write sf object to geojson format
 st_write(df, "tapas_geo.geojson")

"""
# create format with one coordinate per row
coordinates <- df %>%
  st_coordinates() %>%
  as_data_frame() %>%
  rename(`TRIP_ID|integer` = L1)
coordinates$sequence <- ave(c(1:nrow(coordinates)), coordinates$`TRIP_ID|integer`, FUN = seq_along)

# join coordinates with rest of dataframe
long_format <- left_join(coordinates, st_drop_geometry(df), by = c("TRIP_ID|integer"))

write_csv(long_format, "tapas_single_coordinates.csv")
"""

leaflet deckgl in R
mapview in R ist vorgeschaltet

QBIS darstellen

how do i evaluate metrics? if i have something that workss on the original data then i could test if it works on the synthetic data?
Hoe do i evaluate if i have privacy? if someone goes to the strip club every night, how do i make this pattern disappear?
