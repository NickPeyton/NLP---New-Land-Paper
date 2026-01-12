clear all
set more off

cd "D:/PhD/DissolutionProgramming/LND---Land-Paper"
global raw "D:/PhD/DissolutionProgramming/LND---Land-Paper/Data/Raw"
global processed"D:/PhD/DissolutionProgramming/LND---Land-Paper/Data/Processed"

insheet using "$processed/subsidy_master_data.csv"

bysort year recipient_match: sum value, d
bysort year surname_match: sum value, d
