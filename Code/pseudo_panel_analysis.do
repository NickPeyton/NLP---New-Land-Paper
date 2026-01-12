clear all
set more off

cd "D:\PhD\DissolutionProgramming\LND---Land-Paper"

global processed = "Data/Processed"
global raw = "Data/Raw"

use "$processed/hundred_pseudo_panel.dta"

global err_fe "robust absorb(county)"
global err_fe_diff "cluster(par_county) absorb(par_county)"

global ctrls "lLStax_pc area"

global con_1 "longitude latitude cutoff1 cutoff2"
global sfe "county"

global add_y "estadd ysumm"

gen hrv_dum = 0
replace hrv_dum = 1 if lMincome > 0 & lMincome != .

gen nick_dum = 0
replace nick_dum = 1 if llandOwned > 0 & llandOwned != .

gen treat_post = hrv_dum * post
gen nick_treat_post = nick_dum * post

gen hrv_land_post = lMincome * post
gen nick_land_post = llandOwned * post

egen hund_num = group(hundred)
reg max_value_rank_corr post nick_land_post i.hund_num, robust
reg value_rank_corr post nick_land_post i.hund_num, robust
