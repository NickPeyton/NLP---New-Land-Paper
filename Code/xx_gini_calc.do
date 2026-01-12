clear all
set more off

cd "D:/PhD/DissolutionProgramming/LND---Land-Paper"
global raw "D:/PhD/DissolutionProgramming/LND---Land-Paper/Data/Raw"
global processed"D:/PhD/DissolutionProgramming/LND---Land-Paper/Data/Processed"

insheet using "$raw/devon_hundreds_census.csv"
rename Hundreds hundred
drop if hundred == ""
save "$raw/devon_hundreds_census.dta", replace
clear

insheet using "$processed/subsidy_master_data.csv"

egen hund_num = group(hundred)
su hund_num, meanonly

gen pred_gini_1524 = .
gen curve_location_1524 = .
gen curve_scale_1524 = .
gen pred_lower_1524 = .
gen pred_upper_1524 = .
gen pred_gini_censored_1524 = .

replace value = value * 240

forvalues i = 1/`r(max)' {
preserve
quietly: keep if year == 1524 & hund_num == `i' & value != 0 & type == "G"
quietly: capture survlsl value, threshold(240) censorpct(0) model(lognormal)
if _rc == 0 {
	local pred_gini = r(gini)
	local location = r(alpha)
	local scale = r(beta)
	restore
	replace pred_gini_1524 = `pred_gini' if hund_num == `i' & year == 1524
	replace curve_location_1524 = `location' if hund_num == `i' & year == 1524
	replace curve_scale_1524 = `scale' if hund_num == `i' & year == 1524
}
else{
	restore
}
preserve
quietly: keep if year == 1524 & hund_num == `i' & value != 0 & type == "G"
count
local obs = r(N)
local obs1 = `obs' * 42.6
local obs2 = `obs1' / 57.4
local nobs = floor(`obs2')
insobs `nobs'
replace value = 239 if value == .
quietly: capture survbound value, threshold(239) censorpct(.426)
if _rc == 0 {
	local pred_lower = r(lower_a)
	local pred_upper = r(upper_a)
	restore
	replace pred_lower_1524 = `pred_lower' if hund_num == `i' & year == 1524
	replace pred_upper_1524 = `pred_upper' if hund_num == `i' & year == 1524
	replace pred_gini_censored_1524 = (`pred_lower' + `pred_upper') / 2  if hund_num == `i' & year == 1524
}
else {
	restore
}
}
preserve
keep if year == 1524 & type == "G" & value != . & value != 0
survlsl value, threshold(240) censorpct(0) model(lognormal)
count
local obs = r(N)
local obs1 = `obs' * 42.6
local obs2 = `obs1' / 57.4
local nobs = floor(`obs2')
insobs `nobs'
replace value = 239 if value == .
survbound value, threshold(239) censorpct(.426)
restore

gen pred_gini_1543 = .
gen curve_location_1543 = .
gen curve_scale_1543 = .
gen pred_lower_1543 = .
gen pred_upper_1543 = .
gen pred_gini_censored_1543 = .

su hund_num, meanonly
forvalues i = 1/`r(max)' {
preserve
quietly: keep if year == 1543 & hund_num == `i' & value != 0 & type == "G"
quietly: capture survlsl value, threshold(240) censorpct(0) model(lognormal)
if _rc == 0 {
	local pred_gini = r(gini)
	local location = r(alpha)
	local scale = r(beta)
	restore
	replace pred_gini_1543 = `pred_gini' if hund_num == `i' & year == 1543
	replace curve_location_1543 = `location' if hund_num == `i' & year == 1543
	replace curve_scale_1543 = `scale' if hund_num == `i' & year == 1543
}
else{
	restore
}
preserve
quietly: keep if year == 1543 & hund_num == `i' & value != 0 & type == "G"
count
local obs = r(N)
local obs1 = `obs' * 42.6
local obs2 = `obs1' / 57.4
local nobs = floor(`obs2')
insobs `nobs'
replace value = 239 if value == .
quietly: capture survbound value, threshold(239) censorpct(.426)
if _rc == 0 {
	local pred_lower = r(lower_a)
	local pred_upper = r(upper_a)
	restore
	replace pred_lower_1543 = `pred_lower' if hund_num == `i' & year == 1543
	replace pred_upper_1543 = `pred_upper' if hund_num == `i' & year == 1543
	replace pred_gini_censored_1543 = (`pred_lower' + `pred_upper') / 2  if hund_num == `i' & year == 1543
}
else {
	restore
}
}

preserve
keep if year == 1543 & type == "G" & value != . & value != 0
survlsl value, threshold(240) censorpct(0) model(lognormal)
count
local obs = r(N)
local obs1 = `obs' * 42.6
local obs2 = `obs1' / 57.4
local nobs = floor(`obs2')
insobs `nobs'
replace value = 240 if value == .
survbound value, threshold(240) censorpct(.426)
survlsl value, threshold(240) censorpct(.426) model(lognormal)
restore


preserve
collapse pred_gini_censored_1524 pred_gini_censored_1543, by(hundred)
export delimited "$processed/pred_ginis.csv", replace
restore