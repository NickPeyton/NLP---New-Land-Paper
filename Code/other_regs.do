clear all
set more off

cd "D:\PhD\DissolutionProgramming\LND---Land-Paper"

global processed = "Data/Processed"
global raw = "Data/Raw"

insheet using "$processed/hundred_data.csv"

gen cutoff1 = 4
gen cutoff2 = 4
global controls "area mean_elev mean_slope wheatsuit lspc1332 lspc1525 distriver distmkt latitude longitude"

reg _1852_max_value_rank_corr llandowned $controls, robust
reg _1852_value_rank_corr llandowned $controls, robust
