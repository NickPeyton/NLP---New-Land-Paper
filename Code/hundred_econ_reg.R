pacman::p_load(sf, tidyverse, stargazer, sp, dplyr,
               cem, MatchIt, WeightIt, marginaleffects, ipw,
               survey, optmatch, conflicted, cobalt, twang, glue, MASS
)
conflict_prefer('filter', 'dplyr')
conflict_prefer('select', 'dplyr')

setwd('C://PhD/DissolutionProgramming/LND---Land-Paper/')
PROCESSED = 'Data/Processed'
RAW = 'Data/Raw'
SURNAMES = glue('{PROCESSED}/surname_info')
IMAGES = 'Output/Images'
TABLES = 'Output/Tables'

pdf = read.csv(glue('{PROCESSED}/hundred_dataset.csv'), check.names=FALSE)

# If ind_1831 is NA, set agr_1831 to NA
pdf$agr_1831[is.na(pdf$ind_1831)] <- NA

# If ind_1831 == 0 and agr_1831 == 0, set both to NA
pdf$ind_1831[pdf$ind_1831 == 0 & pdf$agr_1831 == 0] <- NA
pdf$agr_1831[pdf$ind_1831 == 0 & pdf$agr_1831 == 0] <- NA

pdf$landOwned[is.na(pdf$landOwned)] <- 0
# Create other_1831 = 1 - ind_1831 - agr_1831
pdf$other_1831 <- 1 - pdf$ind_1831 - pdf$agr_1831

# Log-transform landOwned (+1 to avoid log(0))
pdf$lland <- log(pdf$landOwned + 1)

pdf$monast_land_share <- (pdf$landOwned/240) / (pdf$landOwned/240 + pdf$hundred_val_1524)

# Rename columns: replace "value_1840" with "val_1840"
names(pdf) <- sub("value_1840", "val_1840", names(pdf))

rdf<-pdf
rdf <- rdf %>% filter(!is.na(monast_land_share))
###############################################################################
weightitmodel <- weightit(monast_land_share ~ mean_slope + wheatsuit + 
                            area + lspc1525 + distmkt + hundred_master_treatment_1524 + 
                            hundred_master_control_1524,
                          data=rdf,
                          method='ps'
)
weights <- weightitmodel$weights
ind_weighted_rlm <- rlm(ind_1831 ~ monast_land_share,
                    weights=weights,
                    data=rdf)

print(summary(ind_weighted_rlm))

###############################################################################


weightitmodel <- weightit(monast_land_share ~ mean_slope + wheatsuit + 
                            area + lspc1525 + distmkt + hundred_master_treatment_1524 + 
                            hundred_master_control_1524,
                          data=rdf,
                          method='ps'
)
weights <- weightitmodel$weights
agr_weighted_rlm <- rlm(agr_1831 ~ monast_land_share,
                  weights=weights,
                  data=rdf)

print(summary(agr_weighted_rlm))

###############################################################################
weightitmodel <- weightit(monast_land_share ~ mean_slope + wheatsuit + 
                            area + lspc1525 + distmkt + hundred_master_treatment_1524 + 
                            hundred_master_control_1524,
                          data=rdf,
                          method='ps'
)
weights <- weightitmodel$weights
other_weighted_rlm <- rlm(other_1831 ~ monast_land_share,
                  weights=weights,
                  data=rdf)

print(summary(other_weighted_rlm))


###############################################################################
stargazer(ind_weighted_rlm, agr_weighted_rlm, other_weighted_rlm,
          type = "latex",
          title = "Inverse-Probability-Weighted Regressions",
          align = TRUE,
          table.placement = 'H',
          column.labels = c('Industry', 'Agriculture', 'Other'),
          add.lines = list(c('Geographic Controls', 'Y', 'Y', 'Y')),
          covariate.labels = c('Monastic Land Share'),
          column.sep.width = '.5pt',
          omit.stat = c('aic', 'lr', 'wald', 'logrank'),
          omit = 'Constant',
          out = 'Output/Tables/IPW_hundred.tex'
)


###############################################################################
###############################################################################

rdf<-pdf
rdf <- rdf %>% filter(!is.na(hundred_master_treatment_1524))
weightitmodel <- weightit(hundred_master_treatment_1524 ~ mean_slope + wheatsuit + 
                            area + lspc1525 + distmkt + monast_land_share + 
                            hundred_master_control_1524,
                          data=rdf,
                          method='ps'
)
weights <- weightitmodel$weights
weighted_rlm <- rlm(agr_1831 ~ hundred_master_treatment_1524,
                  weights=weights,
                  data=rdf)

print(summary(weighted_rlm))

###############################################################################
weightitmodel <- weightit(hundred_master_treatment_1524 ~ mean_slope + wheatsuit + 
                            area + lspc1525 + distmkt + monast_land_share + 
                            hundred_master_control_1524,
                          data=rdf,
                          method='ps'
)
weights <- weightitmodel$weights
weighted_rlm <- rlm(ind_1831 ~ hundred_master_treatment_1524,
                  weights=weights,
                  data=rdf)

print(summary(weighted_rlm))

###############################################################################
weightitmodel <- weightit(hundred_master_treatment_1524 ~ mean_slope + wheatsuit + 
                            area + lspc1525 + distmkt + monast_land_share + 
                            hundred_master_control_1524,
                          data=rdf,
                          method='ps'
)
weights <- weightitmodel$weights
weighted_rlm <- rlm(other_1831 ~ hundred_master_treatment_1524,
                  weights=weights,
                  data=rdf)

print(summary(weighted_rlm))

###############################################################################
###############################################################################

rdf<-pdf
rdf <- rdf %>% filter(!is.na(hundred_master_control_1524))
weightitmodel <- weightit(hundred_master_control_1524 ~ mean_slope + wheatsuit + 
                            area + lspc1525 + distmkt + monast_land_share + 
                            hundred_master_treatment_1524,
                          data=rdf,
                          method='ps'
)
weights <- weightitmodel$weights
weighted_rlm <- rlm(agr_1831 ~ hundred_master_control_1524,
                    weights=weights,
                    data=rdf)

print(summary(weighted_rlm))

###############################################################################
weightitmodel <- weightit(hundred_master_control_1524 ~ mean_slope + wheatsuit + 
                            area + lspc1525 + distmkt + monast_land_share + 
                            hundred_master_treatment_1524,
                          data=rdf,
                          method='ps'
)
weights <- weightitmodel$weights
weighted_rlm <- rlm(ind_1831 ~ hundred_master_control_1524,
                    weights=weights,
                    data=rdf)

print(summary(weighted_rlm))

###############################################################################
weightitmodel <- weightit(hundred_master_control_1524 ~ mean_slope + wheatsuit + 
                            area + lspc1525 + distmkt + monast_land_share + 
                            hundred_master_treatment_1524,
                          data=rdf,
                          method='ps'
)
weights <- weightitmodel$weights
weighted_rlm <- rlm(other_1831 ~ hundred_master_control_1524,
                    weights=weights,
                    data=rdf)

print(summary(weighted_rlm))

###############################################################################