clear all
set more off

cd "C:/PhD/DissolutionProgramming/LND---Land-Paper"
global raw "C:/PhD/DissolutionProgramming/LND---Land-Paper/Data/Raw"
global processed "C:/PhD/DissolutionProgramming/LND---Land-Paper/Data/Processed"
global tables "C:/PhD/DissolutionProgramming/LND---Land-Paper/Output/Tables"

insheet using "$processed/hundred_data.csv"
drop if hundred == ""
save "$processed/hundred_data.dta", replace
clear

insheet using "$processed/tithe_landowners.csv"
drop if landowner == ""
merge m:1 hundred using "$processed/hundred_data.dta"


// replace houses = regexreplace(houses, ",", "")
// destring houses, replace
// replace families = regexreplace(families, ",", "")
// destring families, replace
// gen l_cap = regexreplace(cap, "£", "")
// destring l_cap, replace
gen crowding = families/houses

collapse (sum) area_perches (mean) mean_elev mean_slope wheatsuit families llandowned pred_gini_censored_1524 pred_gini_censored_1543 area lspc1525 indag wheatyield distriver distmkt distcoal _1852_value_rank_corr _1852_max_value_rank_corr crowding nrgentry nrpatents, by(hundred landowner)

egen hund_num = group(hundred)
su hund_num, meanonly
gen land_gini = .
gen share_largest = .
gen share_largest_3 = .
gen p99 = .
gen p90 = .
gen p80 = .
gen p70 = .
gen p60 = .
gen p50 = .
gen p40 = .
gen p30 = .
gen p20 = .
gen p10 = .
gen p00 = .
gen landowning_ratio = .

forvalues i = 1/`r(max)' {
preserve
keep if hund_num == `i'
count
display(families)
local obs = r(N)
local extra = families - `obs'
local landowning_ratio = `obs' / families
insobs `extra'
replace area_perches = 0 if area_perches == .
gsort -area_perches
egen tot_perches = total(area_perches)
gen share = area_perches / tot_perches
local share_largest = share in 1
local share_2nd_largest = share in 2
local share_3rd_largest = share in 3
quietly: ineqdeco area_perches
local land_gini = r(gini)

quietly: pshare area_perches, p(99)
local p99 = _b[99-100]
quietly: pshare area_perches, p(10 20 30 40 50 60 70 80 90)
local p90 = _b[90-100]
local p80 = _b[80-90]
local p70 = _b[70-80]
local p60 = _b[60-70]
local p50 = _b[50-60]
local p40 = _b[40-50]
local p30 = _b[30-40]
local p20 = _b[20-30]
local p10 = _b[10-20]
local p00 = _b[0-10]

restore
replace land_gini = `land_gini' if hund_num == `i'
replace share_largest = `share_largest' if hund_num == `i'
replace share_largest_3 = `share_largest' + `share_2nd_largest' + `share_3rd_largest' if hund_num == `i'
replace p99 = `p99' if hund_num == `i'
replace p90 = `p90' if hund_num == `i'
replace p80 = `p80' if hund_num == `i'
replace p70 = `p70' if hund_num == `i'
replace p60 = `p60' if hund_num == `i'
replace p50 = `p50' if hund_num == `i'
replace p40 = `p40' if hund_num == `i'
replace p30 = `p30' if hund_num == `i'
replace p20 = `p20' if hund_num == `i'
replace p10 = `p10' if hund_num == `i'
replace p00 = `p00' if hund_num == `i'
replace landowning_ratio = `landowning_ratio' if hund_num == `i'
}

gen gini_diff = land_gini - pred_gini_censored_1524
collapse (mean) area_perches mean_elev mean_slope wheatsuit families llandowned pred_gini_censored_1524 pred_gini_censored_1543 area lspc1525 land_gini gini_diff indag wheatyield distriver distmkt distcoal _1852_value_rank_corr _1852_max_value_rank_corr share_largest share_largest_3 p90 p99 p80 p70 p60 p50 p40 p30 p20 p10 p00 crowding nrgentry nrpatents landowning_ratio, by(hundred)



eststo clear
reg _1852_value_rank_corr llandowned mean_slope wheatsuit area lspc1525 distmkt , robust
eststo val_rank_corr
reg _1852_max_value_rank_corr llandowned mean_slope wheatsuit area lspc1525 distmkt , robust
eststo max_value_rank_corr

esttab using $tables/surname_mobility.tex, star(* .10 ** .05 *** .01) title(Surname Mobility Regressions) replace

eststo clear
reg land_gini pred_gini_censored_1524 llandowned mean_slope wheatsuit area lspc1525 distmkt , robust
eststo GiniDiff
reg p99 llandowned mean_slope wheatsuit area lspc1525 distmkt , robust
eststo Top1PctShare
reg p90 llandowned mean_slope wheatsuit area lspc1525 distmkt , robust
eststo Top10PctShare
reg landowning_ratio llandowned mean_slope wheatsuit area lspc1525 distmkt , robust
eststo land_ratio
esttab using $tables/inequality.tex, star(* .10 ** .05 *** .01) title(Inequality Regressions) replace

eststo clear
reg nrgentry llandowned mean_slope wheatsuit area lspc1525 distmkt , robust
eststo gentry
reg nrpatents llandowned mean_slope wheatsuit area lspc1525 distmkt , robust
eststo patents

esttab using $tables/gentry_patents.tex, star(* .10 ** .05 *** .01) title(Gentry and Patents Regressions) replace
