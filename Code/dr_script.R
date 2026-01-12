library(tidyverse)
library(dplyr)
library(ggplot2)
library(tmle)
library(glue)
library(SuperLearner)
library(dbarts)
library(xtable)

setwd('C://PhD/DissolutionProgramming/LND---Land-Paper/')
PROCESSED = 'Data/Processed'
RAW = 'Data/Raw'
SURNAMES = glue('{PROCESSED}/surname_info')
IMAGES = 'Output/Images'
TABLES = 'Output/Tables'

df = read_csv(glue('{SURNAMES}/master_id_subsidy_data.csv'))
wills_df <- read_csv(file.path(PROCESSED, "ukda_pcc_wills_final.csv"))

wills_df <- wills_df %>% filter(year<1536)


df <- df %>%
  rowwise() %>%
  mutate(
    will_share_1536 = sum(wills_df[["master_id"]] == id) / nrow(wills_df)
  ) %>%
  ungroup()

df$tot_val_pctile_1524 <- (rank(df$tot_val_1524, ties.method = "average")) / length(df$tot_val_1524)
df$tot_val_pctile_1581 <- (rank(df$tot_val_1581, ties.method = "average")) / length(df$tot_val_1581)
df$tot_val_pctile_1674 <- (rank(df$tot_val_1674, ties.method = "average")) / length(df$tot_val_1674)
df$tot_val_pctile_1840 <- (rank(df$tot_val_1840, ties.method = "average")) / length(df$tot_val_1840)

df$avg_val_pctile_1524 <- (rank(df$avg_val_1524, ties.method = "average")) / length(df$avg_val_1524)
df$avg_val_pctile_1581 <- (rank(df$avg_val_1581, ties.method = "average")) / length(df$avg_val_1581)
df$avg_val_pctile_1674 <- (rank(df$avg_val_1674, ties.method = "average")) / length(df$avg_val_1674)
df$avg_val_pctile_1840 <- (rank(df$avg_val_1840, ties.method = "average")) / length(df$avg_val_1840)

df$max_val_pctile_1524 <- (rank(df$max_val_1524, ties.method = "average")) / length(df$max_val_1524)
df$max_val_pctile_1581 <- (rank(df$max_val_1581, ties.method = "average")) / length(df$max_val_1581)
df$max_val_pctile_1674 <- (rank(df$max_val_1674, ties.method = "average")) / length(df$max_val_1674)
df$max_val_pctile_1840 <- (rank(df$max_val_1840, ties.method = "average")) / length(df$max_val_1840)

df$count_pctile_1524 <- (rank(df$count_1524, ties.method = "average")) / length(df$count_1524)
df$count_pctile_1581 <- (rank(df$count_1581, ties.method = "average")) / length(df$count_1581)
df$count_pctile_1674 <- (rank(df$count_1674, ties.method = "average")) / length(df$count_1674)
df$count_pctile_1840 <- (rank(df$count_1840, ties.method = "average")) / length(df$count_1840)

## Total Surname Value Regressions
# Total Surname Value 1581

Y = df$tot_val_pctile_1581
Y[is.na(Y)] = 0
A = df$treatment
A = as.numeric(A)
A[is.na(A)] = 0
# Get W
W = df %>%
  select(
    will_share_1536,
    ln_max_val_1524,
    ln_parish_avg_value_1524,
    ln_parish_avg_value_1581,
    ln_parish_avg_value_1674,
    ln_parish_monastic_land,
    count_pctile_1524,
    tot_val_pctile_1524,
    tot_val_pctile_1581,
    tot_val_pctile_1674,
    parish_distmkt,
    parish_distriver
  )
W[is.na(W)] = 0
ID = df$id
ID[is.na(ID)] = 0
fit <- tmle(
  Y = Y,
  A = A,
  W = W,
  gform = "A~will_share_1536+tot_val_pctile_1524+count_pctile_1524+ln_max_val_1524+ln_parish_avg_value_1524+ln_parish_monastic_land",
  Qform = "Y~A+tot_val_pctile_1524+parish_distmkt+parish_distriver+ln_parish_avg_value_1524",
)

sfit <- summary(fit)
print(sfit)

est_tot_1581 <- fit$estimates$ATE$psi
se_tot_1581 <- fit$estimates$ATE$var.psi^.5
upper_tot_1581 <- est_tot_1581 + 1.96 * se_tot_1581
lower_tot_1581 <- est_tot_1581 - 1.96 * se_tot_1581
p_tot_1581 <- 2 * (1 - pnorm(abs(est_tot_1581 / se_tot_1581)))

# Total Surname Value 1674

Y = df$tot_val_pctile_1674
Y[is.na(Y)] = 0
A = df$treatment
A = as.numeric(A)
A[is.na(A)] = 0
# Get W
W = df %>%
  select(
    will_share_1536,
    ln_max_val_1524,
    ln_parish_avg_value_1524,
    ln_parish_avg_value_1581,
    ln_parish_avg_value_1674,
    ln_parish_monastic_land,
    count_pctile_1524,
    tot_val_pctile_1524,
    tot_val_pctile_1581,
    tot_val_pctile_1674,
    parish_distmkt,
    parish_distriver
  )
W[is.na(W)] = 0
ID = df$id
ID[is.na(ID)] = 0
fit <- tmle(
  Y = Y,
  A = A,
  W = W,
  gform = "A~will_share_1536+tot_val_pctile_1524+count_pctile_1524+ln_max_val_1524+ln_parish_avg_value_1524+ln_parish_monastic_land",
  Qform = "Y~A+tot_val_pctile_1581+parish_distmkt+parish_distriver+ln_parish_avg_value_1581",
)

sfit <- summary(fit)
print(sfit)

est_tot_1674 <- fit$estimates$ATE$psi
se_tot_1674 <- fit$estimates$ATE$var.psi^.5
upper_tot_1674 <- est_tot_1674 + 1.96 * se_tot_1674
lower_tot_1674 <- est_tot_1674 - 1.96 * se_tot_1674
p_tot_1674 <- 2 * (1 - pnorm(abs(est_tot_1674 / se_tot_1674)))


# Total Surname Value 1840
Y = df$tot_val_pctile_1840
Y[is.na(Y)] = 0
A = df$treatment
A = as.numeric(A)
A[is.na(A)] = 0
# Get W
W = df %>%
  select(
    will_share_1536,
    ln_max_val_1524,
    ln_parish_avg_value_1524,
    ln_parish_avg_value_1581,
    ln_parish_avg_value_1674,
    ln_parish_monastic_land,
    count_pctile_1524,
    tot_val_pctile_1524,
    tot_val_pctile_1581,
    tot_val_pctile_1674,
    parish_distmkt,
    parish_distriver
  )
W[is.na(W)] = 0
ID = df$id
ID[is.na(ID)] = 0
fit <- tmle(
  Y = Y,
  A = A,
  W = W,
  gform = "A~will_share_1536+tot_val_pctile_1524+count_pctile_1524+ln_max_val_1524+ln_parish_avg_value_1524+ln_parish_monastic_land",
  Qform = "Y~A+tot_val_pctile_1674+parish_distmkt+parish_distriver+ln_parish_avg_value_1674",
  )

sfit <- summary(fit)
print(sfit)

est_tot_1840 <- fit$estimates$ATE$psi
se_tot_1840 <- fit$estimates$ATE$var.psi^.5
upper_tot_1840 <- est_tot_1840 + 1.96 * se_tot_1840
lower_tot_1840 <- est_tot_1840 - 1.96 * se_tot_1840
p_tot_1840 <- 2 * (1 - pnorm(abs(est_tot_1840 / se_tot_1840)))

tot_val_results <- data.frame('var' = c('Estimate', 'SE', 'N'),
  '1581' = c(est_tot_1581, se_tot_1581, length(Y)),
  '1674' = c(est_tot_1674, se_tot_1674, length(Y)),
  '1840' = c(est_tot_1840, se_tot_1840, length(Y))
  )
colnames(tot_val_results) <- c("var", "1581", "1674", "1840")
tot_val_results <- tot_val_results %>%
  column_to_rownames('var')
## Average Surname Value Regressions
# Average Surname Value 1581

Y = df$avg_val_pctile_1581
Y[is.na(Y)] = 0
A = df$treatment
A = as.numeric(A)
A[is.na(A)] = 0
# Get W
W = df %>%
  select(
    will_share_1536,
    ln_max_val_1524,
    ln_parish_avg_value_1524,
    ln_parish_avg_value_1581,
    ln_parish_avg_value_1674,
    ln_parish_monastic_land,
    avg_val_pctile_1524,
    avg_val_pctile_1581,
    avg_val_pctile_1674,
    parish_distmkt,
    count_pctile_1524,
    tot_val_pctile_1524,
    parish_distriver
  )
W[is.na(W)] = 0
ID = df$id
ID[is.na(ID)] = 0
fit <- tmle(
  Y = Y,
  A = A,
  W = W,
  gform = "A~will_share_1536+tot_val_pctile_1524+count_pctile_1524+ln_max_val_1524+ln_parish_avg_value_1524+ln_parish_monastic_land",
  Qform = "Y~A+avg_val_pctile_1524+parish_distmkt+parish_distriver+ln_parish_avg_value_1524",
)

sfit <- summary(fit)
print(sfit)

est_avg_1581 <- fit$estimates$ATE$psi
se_avg_1581 <- fit$estimates$ATE$var.psi^.5
upper_avg_1581 <- est_avg_1581 + 1.96 * se_avg_1581
lower_avg_1581 <- est_avg_1581 - 1.96 * se_avg_1581
p_avg_1581 <- 2 * (1 - pnorm(abs(est_avg_1581 / se_avg_1581)))


# Average Surname Value 1674

Y = df$avg_val_pctile_1674
Y[is.na(Y)] = 0
A = df$treatment
A = as.numeric(A)
A[is.na(A)] = 0
# Get W
W = df %>%
  select(
    will_share_1536,
    ln_max_val_1524,
    ln_parish_avg_value_1524,
    ln_parish_avg_value_1581,
    ln_parish_avg_value_1674,
    ln_parish_monastic_land,
    avg_val_pctile_1524,
    avg_val_pctile_1581,
    avg_val_pctile_1674,
    parish_distmkt,
    count_pctile_1524,
    tot_val_pctile_1524,
    parish_distriver
  )
W[is.na(W)] = 0
ID = df$id
ID[is.na(ID)] = 0
fit <- tmle(
  Y = Y,
  A = A,
  W = W,
  gform = "A~will_share_1536+tot_val_pctile_1524+count_pctile_1524+ln_max_val_1524+ln_parish_avg_value_1524+ln_parish_monastic_land",
  Qform = "Y~A+avg_val_pctile_1581+parish_distmkt+parish_distriver+ln_parish_avg_value_1581",
)

sfit <- summary(fit)
print(sfit)

est_avg_1674 <- fit$estimates$ATE$psi
se_avg_1674 <- fit$estimates$ATE$var.psi^.5
upper_avg_1674 <- est_avg_1674 + 1.96 * se_avg_1674
lower_avg_1674 <- est_avg_1674 - 1.96 * se_avg_1674
p_avg_1674 <- 2 * (1 - pnorm(abs(est_avg_1674 / se_avg_1674)))


# Average Surname Value 1840
Y = df$avg_val_pctile_1840
Y[is.na(Y)] = 0
A = df$treatment
A = as.numeric(A)
A[is.na(A)] = 0
# Get W
W = df %>%
  select(
    will_share_1536,
    ln_max_val_1524,
    ln_parish_avg_value_1524,
    ln_parish_avg_value_1581,
    ln_parish_avg_value_1674,
    ln_parish_monastic_land,
    avg_val_pctile_1524,
    avg_val_pctile_1581,
    avg_val_pctile_1674,
    parish_distmkt,
    count_pctile_1524,
    tot_val_pctile_1524,
    parish_distriver
  )
W[is.na(W)] = 0
ID = df$id
ID[is.na(ID)] = 0
fit <- tmle(
  Y = Y,
  A = A,
  W = W,
  gform = "A~will_share_1536+tot_val_pctile_1524+count_pctile_1524+ln_max_val_1524+ln_parish_avg_value_1524+ln_parish_monastic_land",
  Qform = "Y~A+avg_val_pctile_1674+parish_distmkt+parish_distriver+ln_parish_avg_value_1674",
)

sfit <- summary(fit)
print(sfit)

est_avg_1840 <- fit$estimates$ATE$psi
se_avg_1840 <- fit$estimates$ATE$var.psi^.5
upper_avg_1840 <- est_avg_1840 + 1.96 * se_avg_1840
lower_avg_1840 <- est_avg_1840 - 1.96 * se_avg_1840
p_avg_1840 <- 2 * (1 - pnorm(abs(est_avg_1840 / se_avg_1840)))

avg_val_results <- data.frame('var' = c('Estimate', 'SE', 'N'),
                              '1581' = c(est_avg_1581, se_avg_1581, length(Y)),
                              '1674' = c(est_avg_1674, se_avg_1674, length(Y)),
                              '1840' = c(est_avg_1840, se_avg_1840, length(Y))
)
colnames(avg_val_results) <- c("var", "1581", "1674", "1840")
avg_val_results <- avg_val_results %>%
  column_to_rownames('var')


## Maximum Surname Value Regressions
# Maximum Surname Value 1581

Y = df$max_val_pctile_1581
Y[is.na(Y)] = 0
A = df$treatment
A = as.numeric(A)
A[is.na(A)] = 0
# Get W
W = df %>%
  select(
    will_share_1536,
    ln_max_val_1524,
    ln_parish_avg_value_1524,
    ln_parish_avg_value_1581,
    ln_parish_avg_value_1674,
    ln_parish_monastic_land,
    max_val_pctile_1524,
    max_val_pctile_1581,
    max_val_pctile_1674,
    parish_distmkt,
    count_pctile_1524,
    tot_val_pctile_1524,
    parish_distriver
  )
W[is.na(W)] = 0
ID = df$id
ID[is.na(ID)] = 0
fit <- tmle(
  Y = Y,
  A = A,
  W = W,
  gform = "A~will_share_1536+tot_val_pctile_1524+count_pctile_1524+ln_max_val_1524+ln_parish_avg_value_1524+ln_parish_monastic_land",
  Qform = "Y~A+max_val_pctile_1524+parish_distmkt+parish_distriver+ln_parish_avg_value_1524",
)

sfit <- summary(fit)
print(sfit)

est_max_1581 <- fit$estimates$ATE$psi
se_max_1581 <- fit$estimates$ATE$var.psi^.5
upper_max_1581 <- est_max_1581 + 1.96 * se_max_1581
lower_max_1581 <- est_max_1581 - 1.96 * se_max_1581
p_max_1581 <- 2 * (1 - pnorm(abs(est_max_1581 / se_max_1581)))


# Maximum Surname Value 1674

Y = df$max_val_pctile_1674
Y[is.na(Y)] = 0
A = df$treatment
A = as.numeric(A)
A[is.na(A)] = 0
# Get W
W = df %>%
  select(
    will_share_1536,
    ln_max_val_1524,
    ln_parish_avg_value_1524,
    ln_parish_avg_value_1581,
    ln_parish_avg_value_1674,
    ln_parish_monastic_land,
    max_val_pctile_1524,
    max_val_pctile_1581,
    max_val_pctile_1674,
    parish_distmkt,
    count_pctile_1524,
    tot_val_pctile_1524,
    parish_distriver
  )
W[is.na(W)] = 0
ID = df$id
ID[is.na(ID)] = 0
fit <- tmle(
  Y = Y,
  A = A,
  W = W,
  gform = "A~will_share_1536+tot_val_pctile_1524+count_pctile_1524+ln_max_val_1524+ln_parish_avg_value_1524+ln_parish_monastic_land",
  Qform = "Y~A+max_val_pctile_1581+parish_distmkt+parish_distriver+ln_parish_avg_value_1581",
)

sfit <- summary(fit)
print(sfit)

est_max_1674 <- fit$estimates$ATE$psi
se_max_1674 <- fit$estimates$ATE$var.psi^.5
upper_max_1674 <- est_max_1674 + 1.96 * se_max_1674
lower_max_1674 <- est_max_1674 - 1.96 * se_max_1674
p_max_1674 <- 2 * (1 - pnorm(abs(est_max_1674 / se_max_1674)))


# Maximum Surname Value 1840
Y = df$max_val_pctile_1840
Y[is.na(Y)] = 0
A = df$treatment
A = as.numeric(A)
A[is.na(A)] = 0
# Get W
W = df %>%
  select(
    will_share_1536,
    ln_max_val_1524,
    ln_parish_avg_value_1524,
    ln_parish_avg_value_1581,
    ln_parish_avg_value_1674,
    ln_parish_monastic_land,
    max_val_pctile_1524,
    max_val_pctile_1581,
    max_val_pctile_1674,
    parish_distmkt,
    count_pctile_1524,
    tot_val_pctile_1524,
    parish_distriver
  )
W[is.na(W)] = 0
ID = df$id
ID[is.na(ID)] = 0
fit <- tmle(
  Y = Y,
  A = A,
  W = W,
  gform = "A~will_share_1536+tot_val_pctile_1524+count_pctile_1524+ln_max_val_1524+ln_parish_avg_value_1524+ln_parish_monastic_land",
  Qform = "Y~A+max_val_pctile_1674+parish_distmkt+parish_distriver+ln_parish_avg_value_1674",
)

sfit <- summary(fit)
print(sfit)

est_max_1840 <- fit$estimates$ATE$psi
se_max_1840 <- fit$estimates$ATE$var.psi^.5
upper_max_1840 <- est_max_1840 + 1.96 * se_max_1840
lower_max_1840 <- est_max_1840 - 1.96 * se_max_1840
p_max_1840 <- 2 * (1 - pnorm(abs(est_max_1840 / se_max_1840)))

max_val_results <- data.frame('var' = c('Estimate', 'SE', 'N'),
                              '1581' = c(est_max_1581, se_max_1581, length(Y)),
                              '1674' = c(est_max_1674, se_max_1674, length(Y)),
                              '1840' = c(est_max_1840, se_max_1840, length(Y))
)
colnames(max_val_results) <- c("var", "1581", "1674", "1840")
max_val_results <- max_val_results %>%
  column_to_rownames('var')


## Surname Count Regressions
# Surname Count 1581

Y = df$count_pctile_1581
Y[is.na(Y)] = 0
A = df$treatment
A = as.numeric(A)
A[is.na(A)] = 0
# Get W
W = df %>%
  select(
    will_share_1536,
    ln_max_val_1524,
    ln_parish_avg_value_1524,
    ln_parish_avg_value_1581,
    ln_parish_avg_value_1674,
    ln_parish_monastic_land,
    count_pctile_1524,
    count_pctile_1581,
    count_pctile_1674,
    parish_distmkt,
    count_pctile_1524,
    tot_val_pctile_1524,
    parish_distriver
  )
W[is.na(W)] = 0
ID = df$id
ID[is.na(ID)] = 0
fit <- tmle(
  Y = Y,
  A = A,
  W = W,
  gform = "A~will_share_1536+tot_val_pctile_1524+count_pctile_1524+ln_max_val_1524+ln_parish_avg_value_1524+ln_parish_monastic_land",
  Qform = "Y~A+count_pctile_1524+parish_distmkt+parish_distriver+ln_parish_avg_value_1524",
)

sfit <- summary(fit)
print(sfit)

est_count_1581 <- fit$estimates$ATE$psi
se_count_1581 <- fit$estimates$ATE$var.psi^.5
upper_count_1581 <- est_count_1581 + 1.96 * se_count_1581
lower_count_1581 <- est_count_1581 - 1.96 * se_count_1581
p_count_1581 <- 2 * (1 - pnorm(abs(est_count_1581 / se_count_1581)))


# Surname Count 1674

Y = df$count_pctile_1674
Y[is.na(Y)] = 0
A = df$treatment
A = as.numeric(A)
A[is.na(A)] = 0
# Get W
W = df %>%
  select(
    will_share_1536,
    ln_max_val_1524,
    ln_parish_avg_value_1524,
    ln_parish_avg_value_1581,
    ln_parish_avg_value_1674,
    ln_parish_monastic_land,
    count_pctile_1524,
    count_pctile_1581,
    count_pctile_1674,
    parish_distmkt,
    count_pctile_1524,
    tot_val_pctile_1524,
    parish_distriver
  )
W[is.na(W)] = 0
ID = df$id
ID[is.na(ID)] = 0
fit <- tmle(
  Y = Y,
  A = A,
  W = W,
  gform = "A~will_share_1536+tot_val_pctile_1524+count_pctile_1524+ln_max_val_1524+ln_parish_avg_value_1524+ln_parish_monastic_land",
  Qform = "Y~A+count_pctile_1581+parish_distmkt+parish_distriver+ln_parish_avg_value_1581",
)

sfit <- summary(fit)
print(sfit)

est_count_1674 <- fit$estimates$ATE$psi
se_count_1674 <- fit$estimates$ATE$var.psi^.5
upper_count_1674 <- est_count_1674 + 1.96 * se_count_1674
lower_count_1674 <- est_count_1674 - 1.96 * se_count_1674
p_count_1674 <- 2 * (1 - pnorm(abs(est_count_1674 / se_count_1674)))


# Surname Count 1840
Y = df$count_pctile_1840
Y[is.na(Y)] = 0
A = df$treatment
A = as.numeric(A)
A[is.na(A)] = 0
# Get W
W = df %>%
  select(
    will_share_1536,
    ln_max_val_1524,
    ln_parish_avg_value_1524,
    ln_parish_avg_value_1581,
    ln_parish_avg_value_1674,
    ln_parish_monastic_land,
    count_pctile_1524,
    count_pctile_1581,
    count_pctile_1674,
    parish_distmkt,
    count_pctile_1524,
    tot_val_pctile_1524,
    parish_distriver
  )
W[is.na(W)] = 0
ID = df$id
ID[is.na(ID)] = 0
fit <- tmle(
  Y = Y,
  A = A,
  W = W,
  gform = "A~will_share_1536+tot_val_pctile_1524+count_pctile_1524+ln_max_val_1524+ln_parish_avg_value_1524+ln_parish_monastic_land",
  Qform = "Y~A+count_pctile_1674+parish_distmkt+parish_distriver+ln_parish_avg_value_1674",
)

sfit <- summary(fit)
print(sfit)

est_count_1840 <- fit$estimates$ATE$psi
se_count_1840 <- fit$estimates$ATE$var.psi^.5
upper_count_1840 <- est_count_1840 + 1.96 * se_count_1840
lower_count_1840 <- est_count_1840 - 1.96 * se_count_1840
p_count_1840 <- 2 * (1 - pnorm(abs(est_count_1840 / se_count_1840)))

count_results <- data.frame('var' = c('Estimate', 'SE', 'N'),
                              '1581' = c(est_count_1581, se_count_1581, length(Y)),
                              '1674' = c(est_count_1674, se_count_1674, length(Y)),
                              '1840' = c(est_count_1840, se_count_1840, length(Y))
)
colnames(count_results) <- c("var", "1581", "1674", "1840")
count_results <- count_results %>%
  column_to_rownames('var')



## Export all the tables
# Total
tot_xtable = xtable(tot_val_results,
                    caption = 'Total Value Results',
                    label = 'tab:tot_val_dr',
                    align = 'llll')
write.csv(t(tot_val_results), glue('{TABLES}/tot_val_dr.csv'))
print(tot_xtable, 
      type='latex',
      file=glue('{TABLES}/tot_val_dr.tex'),
      include_rownames=TRUE,
      booktabs=TRUE)

# Average
avg_xtable = xtable(avg_val_results,
                    caption = 'Average Value Results',
                    label = 'tab:avg_val_dr',
                    align = 'llll')
write.csv(t(avg_val_results), glue('{TABLES}/avg_val_dr.csv'))
print(avg_xtable, 
      type='latex',
      file=glue('{TABLES}/avg_val_dr.tex'),
      include_rownames=TRUE,
      booktabs=TRUE)

# Maximum
max_xtable = xtable(max_val_results,
                    caption = 'Max Value Results',
                    label = 'tab:max_val_dr',
                    align = 'llll')
write.csv(t(max_val_results), glue('{TABLES}/max_val_dr.csv'))
print(max_xtable, 
      type='latex',
      file=glue('{TABLES}/max_val_dr.tex'),
      include_rownames=TRUE,
      booktabs=TRUE)

# Count
max_xtable = xtable(count_results,
                    caption = 'Count Results',
                    label = 'tab:count_dr',
                    align = 'llll')
write.csv(t(count_results), glue('{TABLES}/count_dr.csv'))
print(max_xtable, 
      type='latex',
      file=glue('{TABLES}/count_dr.tex'),
      include_rownames=TRUE,
      booktabs=TRUE)