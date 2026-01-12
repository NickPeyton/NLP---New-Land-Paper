clear all
set more off

cd "D:\PhD\DissolutionProgramming\LND---Land-Paper"

global processed = "Data/Processed"
global raw = "Data/Raw"

use "$processed/pseudo_panel.dta"

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

{ //// Panel Value Regs 

// // Copyhold Regs
// preserve
// space_reg $con_1 copys hrv_land_post post if copyhold_reg_ind==1, xreg(2) coord(2) model(areg, par_county) 
// reghdfe copys post hrv_land_post if copyhold_reg_ind==1,  cluster(par_county) absorb(par_county)
// restore
//
// preserve
// space_reg $con_1 copys hrv_land_post post if copyhold_reg_ind==1 & county_complete == 1, xreg(2) coord(2) model(areg, county) 
// reghdfe copys post hrv_land_post if copyhold_reg_ind==1 & county_complete == 1,  cluster(county) absorb(county)
// restore
// preserve
// space_reg $con_1 copys nick_land_post post if copyhold_reg_ind==1 & county_complete == 1, xreg(2) coord(2) model(areg, county) 
// reghdfe copys post nick_land_post if copyhold_reg_ind==1 & county_complete == 1,  cluster(county) absorb(county)
// restore

// Market Regs
preserve
space_reg $con_1 market_1600 hrv_land_post post, xreg(2) coord(2) model(areg, par_county) 
reghdfe market_1600 post hrv_land_post,  $err_fe_diff
restore

preserve
space_reg $con_1 market_1600 hrv_land_post post if county_complete == 1, xreg(2) coord(2) model(areg, par_county) 
reghdfe market_1600 post hrv_land_post if county_complete == 1,  $err_fe_diff
restore

preserve
space_reg $con_1 market_1600 nick_land_post post if county_complete == 1, xreg(2) coord(2) model(areg, par_county) 
reghdfe market_1600 post nick_land_post if county_complete == 1,  $err_fe_diff
restore

// Gentry Regs
preserve
space_reg $con_1 NrGentry hrv_land_post post if gentry_reg_ind==1, xreg(2) coord(2) model(areg, par_county) 
reghdfe NrGentry post hrv_land_post if gentry_reg_ind==1,  $err_fe_diff
restore

preserve
space_reg $con_1 NrGentry hrv_land_post post if gentry_reg_ind==1 & county_complete == 1, xreg(2) coord(2) model(areg, par_county) 
reghdfe NrGentry post hrv_land_post if gentry_reg_ind==1 & county_complete == 1,  $err_fe_diff
restore

preserve
space_reg $con_1 NrGentry nick_land_post post if gentry_reg_ind==1 & county_complete == 1, xreg(2) coord(2) model(areg, par_county) 
reghdfe NrGentry post nick_land_post if gentry_reg_ind==1 & county_complete == 1,  $err_fe_diff
restore

// Catholic Regs
preserve
space_reg $con_1 perc_cath hrv_land_post post if cath_1800_ind==1, xreg(2) coord(2) model(areg, par_county) 
reghdfe perc_cath post hrv_land_post if cath_1800_ind==1,  $err_fe_diff
restore

preserve
space_reg $con_1 perc_cath hrv_land_post post if cath_1800_ind==1 & county_complete == 1, xreg(2) coord(2) model(areg, par_county) 
reghdfe perc_cath post hrv_land_post if cath_1800_ind==1 & county_complete == 1,  $err_fe_diff
restore

preserve
space_reg $con_1 perc_cath nick_land_post post if cath_1800_ind==1 & county_complete == 1, xreg(2) coord(2) model(areg, par_county) 
reghdfe perc_cath post nick_land_post if cath_1800_ind==1 & county_complete == 1,  $err_fe_diff
restore

// Mill Regs
preserve
space_reg $con_1 mills hrv_land_post post if mill_reg_ind==1, xreg(2) coord(2) model(areg, par_county) 
reghdfe mills post hrv_land_post if mill_reg_ind==1,  $err_fe_diff
restore

preserve
space_reg $con_1 mills hrv_land_post post if mill_reg_ind==1 & county_complete == 1, xreg(2) coord(2) model(areg, par_county) 
reghdfe mills post hrv_land_post if mill_reg_ind==1 & county_complete == 1,  $err_fe_diff
restore

preserve
space_reg $con_1 mills nick_land_post post if mill_reg_ind==1 & county_complete == 1, xreg(2) coord(2) model(areg, par_county) 
reghdfe mills post nick_land_post if mill_reg_ind==1 & county_complete == 1,  $err_fe_diff
restore

// AgShare Regs
preserve
space_reg $con_1 agr_share hrv_land_post post if agr_reg_ind==1, xreg(2) coord(2) model(areg, par_county) 
reghdfe agr_share post hrv_land_post if agr_reg_ind==1,  cluster(county) absorb(county)
restore

preserve
space_reg $con_1 agr_share hrv_land_post post if agr_reg_ind==1 & county_complete == 1, xreg(2) coord(2) model(areg, par_county) 
reghdfe agr_share post hrv_land_post if agr_reg_ind==1 & county_complete == 1,  cluster(county) absorb(county)
restore

preserve
space_reg $con_1 agr_share nick_land_post post if agr_reg_ind==1 & county_complete == 1, xreg(2) coord(2) model(areg, par_county) 
reghdfe agr_share post nick_land_post if agr_reg_ind==1 & county_complete == 1,  cluster(county) absorb(county)
restore

// IndShare Regs
preserve
space_reg $con_1 ind_share hrv_land_post post if ind_reg_ind==1, xreg(2) coord(2) model(areg, par_county) 
reghdfe ind_share post hrv_land_post if ind_reg_ind==1,  cluster(county) absorb(county)
restore

preserve
space_reg $con_1 ind_share hrv_land_post post if ind_reg_ind==1 & county_complete == 1, xreg(2) coord(2) model(areg, par_county) 
reghdfe ind_share post hrv_land_post if ind_reg_ind==1 & county_complete == 1,  cluster(county) absorb(county)
restore

preserve
space_reg $con_1 ind_share nick_land_post post if ind_reg_ind==1 & county_complete == 1, xreg(2) coord(2) model(areg, par_county) 
reghdfe ind_share post nick_land_post if ind_reg_ind==1 & county_complete == 1,  cluster(county) absorb(county)
restore
}

{ //// Normal Regs

keep if post == 1

// Patents
preserve
space_reg $con_1 NrPatents lMincome $ctrls, xreg(3) coord(2) model(areg, $sfe) 
areg NrPatents lMincome $ctrls,  $err_fe
restore

preserve
space_reg $con_1 NrPatents lMincome $ctrls if county_complete == 1, xreg(3) coord(2) model(areg, $sfe) 
areg NrPatents lMincome $ctrls,  $err_fe
restore

preserve
space_reg $con_1 NrPatents llandOwned $ctrls if county_complete == 1, xreg(3) coord(2) model(areg, $sfe) 
areg NrPatents llandOwned $ctrls,  $err_fe
restore

// Enclosure
preserve
space_reg $con_1 enclosed lMincome $ctrls, xreg(3) coord(2) model(areg, $sfe) 
areg enclosed lMincome $ctrls,  $err_fe
restore
preserve
space_reg $con_1 enclosed lMincome $ctrls if county_complete == 1, xreg(3) coord(2) model(areg, $sfe) 
areg enclosed lMincome $ctrls,  $err_fe
restore
preserve
space_reg $con_1 enclosed llandOwned $ctrls if county_complete == 1, xreg(3) coord(2) model(areg, $sfe) 
areg enclosed llandOwned $ctrls,  $err_fe
restore

// Threshing Machines
preserve
space_reg $con_1 thresh_machines lMincome $ctrls, xreg(3) coord(2) model(areg, $sfe) 
areg thresh_machines lMincome $ctrls,  $err_fe
restore
preserve
space_reg $con_1 thresh_machines lMincome $ctrls if county_complete == 1, xreg(3) coord(2) model(areg, $sfe) 
areg thresh_machines lMincome $ctrls,  $err_fe
restore
preserve
space_reg $con_1 thresh_machines llandOwned $ctrls if county_complete == 1, xreg(3) coord(2) model(areg, $sfe) 
areg thresh_machines llandOwned $ctrls,  $err_fe
restore
// Wheat Yield
preserve
space_reg $con_1 WheatYield lMincome $ctrls, xreg(3) coord(2) model(areg, $sfe) 
areg WheatYield lMincome $ctrls,  $err_fe
restore
preserve
space_reg $con_1 WheatYield lMincome $ctrls if county_complete == 1, xreg(3) coord(2) model(areg, $sfe) 
areg WheatYield lMincome $ctrls,  $err_fe
restore
preserve
space_reg $con_1 WheatYield llandOwned $ctrls if county_complete == 1, xreg(3) coord(2) model(areg, $sfe) 
areg WheatYield llandOwned $ctrls,  $err_fe
restore
}