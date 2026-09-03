



combined <- read.csv("combined_data_treatments.csv")
outcomes_all <- data.frame()
cat_long_all <- data.frame()
outcomes_ind_all <- data.frame()
hr_df <- data.frame()
for (f in 1:5){
predictions <- read.csv(paste0("multitask_model_test_outcomes_fold", f, "_FINAL.csv"))
train_predictions <- read.csv(paste0("multitask_model_train_outcomes_fold", f, "_FINAL.csv"))
val_predictions <- read.csv(paste0("multitask_model_val_outcomes_fold", f, "_FINAL.csv"))

ids <- read.csv(paste0("~//Final Datasets//fold", f, "_ids.csv"))

train_y <- read.csv(paste0("~//Final Datasets//survival_group_fold", f, "_train.csv"))
val_y <- read.csv(paste0("~//Final Datasets//survival_group_fold", f, "_val.csv"))
test_y <- read.csv(paste0("~//Final Datasets//survival_group_fold", f, "_test.csv"))


## get frequency of treated drugs per fold
freq_all <- data.frame()
i <- 1
for (fold in list(train_y, val_y, test_y)){
  for (drug in c("Hormone Therapy",
                 "CDK Inhibitor", "Immunotherapy", "Chemotherapy")){
    fold_t <- fold[,grep(drug, colnames(fold))]
    num <- nrow(na.omit(fold_t))
    freq_all <- rbind(freq_all, data.frame("Fold" = paste("Fold", i, sep = " "),
                                           "Treatment" = drug,
                                           "Count" = num))
  }
  i <- i + 1
}

ggplot(freq_all, aes(x = Fold, y = Count, fill = Treatment)) +
  geom_bar(stat = "identity") +
  theme_classic() +
  scale_fill_observable()



survival <- read.csv("~//Documents//COH Breast Cohort//patient_treatment_responses_time_to_change_FINAL.csv")
survival$Medication_Combination[grep("Hormone", survival$Medication_Combination)] <- "Hormone Therapy"
survival$Medication_Combination[grep("Chemotherapy", survival$Medication_Combination)] <- "Chemotherapy"
survival$Medication_Combination[survival$Medication_Combination == "Immunotherapy (CKI)"] <- "Immunotherapy"

cat <- data.frame(matrix("Non-responder", ncol = ncol(predictions), nrow = nrow(predictions)))
colnames(cat) <- c("Hormone Therapy", "Immunotherapy",
                   "CDK Inhibitor", "Chemotherapy")

library(survcomp)
library(survminer)

find_cutpoint_target_hr <- function(df, val_df,
                                    time_col, event_col, cont_col,
                                    trt_col,
                                    target_logHR = -0.5,
                                    grid_size = 500,
                                    chemo_resp_rate = chemo_rate,
                                    hormone_resp_rate = hormone_rate,
                                    cdk_resp_rate = cdk_rate,
                                    immune_resp_rate = immune_rate) {

  # -------- Clean data --------


  x <- df[[cont_col]]

  # -------- Candidate cutpoints --------
  cutpoints <- seq(min(x, na.rm = TRUE),
                   max(x, na.rm = TRUE),
                   length.out = grid_size)


  results <- data.frame()


  km <- survfit(Surv(df[[time_col]], df[[event_col]]) ~ 1)
  km_6 <- summary(km, times = 365)

  target_prop <- km_6$surv


  for (cp in cutpoints) {

    # =========================
    # --- TRAIN ---
    # =========================
    df$group <- ifelse(df[[cont_col]] <= cp, 0, 1)
    if (length(unique(df$group)) < 2) next
    tab <- table(df$group)
    if (any(tab < 2)) next


    fit_train <- try(
      coxph(Surv(df[[time_col]], df[[event_col]]) ~ group, data = df),
      silent = TRUE
    )
    if (inherits(fit_train, "try-error")) next

    logHR_train <- coef(fit_train)["group"]

    # --- determine responder group (IMPORTANT) ---
    responder_group <- ifelse(logHR_train < 0, 1, 0)

    # --- responder proportion (TRAIN) ---
    prop_resp <- mean(df$group == responder_group)
    tolerance = 0.05
    # --- HARD constraint (TRAIN) ---
    if (prop_resp > (target_prop + tolerance)) next


    # =========================
    # --- VALIDATION ---
    # =========================
    val_df$group <- ifelse(val_df[[cont_col]] <= cp, 0, 1)
    if (length(unique(val_df$group)) < 2) next

    fit_val <- try(
      coxph(Surv(val_df[[time_col]], val_df[[event_col]]) ~ group, data = val_df),
      silent = TRUE
    )
    if (inherits(fit_val, "try-error")) next

    logHR_val <- coef(fit_val)["group"]

    # --- responder proportion (VALIDATION) ---
    prop_resp_val <- mean(val_df$group == responder_group)

    # --- HARD constraint (VALIDATION) ---

    #if (prop_resp_val > (target_prop + tolerance)) next


    # =========================
    # --- SCORE ---
    # =========================

    # Penalize only if validation is weaker than train
    val_penalty <- max(0, logHR_val - logHR_train)

    score <- abs(logHR_train - target_logHR) +
      val_penalty


    # =========================
    # --- STORE RESULTS ---
    # =========================
    results <- rbind(results, data.frame(
      cutpoint = cp,
      logHR_train = logHR_train,
      logHR_val = logHR_val,
      prop_resp = prop_resp,
      prop_resp_val = prop_resp_val,
      score = score
    ))
  }

  # =========================
  # Safety check
  # =========================
  if (nrow(results) == 0) return(NA)

  # Rank by score
  results <- results[order(results$score), ]

  best <- results[1, ]

  return(list(
    best_cutpoint = best$cutpoint,
    best_logHR_train = best$logHR_train,
    best_logHR_val = best$logHR_val,
    results = results
  ))
}



train_predictions <- cbind(train_predictions, train_y)
val_predictions <- cbind(val_predictions, val_y)

hr = 2

result <- find_cutpoint_target_hr(
  df = na.omit(train_predictions[,c("hormone", "Hormone_Time", "Hormone_Event")]),
  val_df = na.omit(val_predictions[,c("hormone", "Hormone_Time", "Hormone_Event")]),
  time_col = "Hormone_Time",
  event_col = "Hormone_Event",
  cont_col = "hormone",
  trt_col = "hormone",
  target_logHR = hr
)

hormone_cp <- result$best_cutpoint
hormone_train_hr <- result$best_logHR_train
hormone_val_hr <- result$best_logHR_val

result <- find_cutpoint_target_hr(
  df = na.omit(train_predictions[,c("cdk", "CDK_Time", "CDK_Event")]),
  val_df = na.omit(val_predictions[,c("cdk", "CDK_Time", "CDK_Event")]),
  time_col = "CDK_Time",
  event_col = "CDK_Event",
  cont_col = "cdk",
  trt_col = "cdk",
  target_logHR = hr
)
cdk_cp <- result$best_cutpoint
cdk_train_hr <- result$best_logHR_train
cdk_val_hr <- result$best_logHR_val

result <- find_cutpoint_target_hr(
  df = na.omit(train_predictions[,c("immune", "Immunotherapy_Time", "Immunotherapy_Event")]),
  val_df = na.omit(val_predictions[,c("immune", "Immunotherapy_Time", "Immunotherapy_Event")]),
  time_col = "Immunotherapy_Time",
  event_col = "Immunotherapy_Event",
  cont_col = "immune",
  trt_col = "immune",
  target_logHR = hr
)
immune_cp <- result$best_cutpoint
immune_train_hr <- result$best_logHR_train
immune_val_hr <- result$best_logHR_val

result <- find_cutpoint_target_hr(
  df = na.omit(train_predictions[,c("chemo", "Chemotherapy_Time", "Chemotherapy_Event")]),
  val_df = na.omit(val_predictions[,c("chemo", "Chemotherapy_Time", "Chemotherapy_Event")]),
  time_col = "Chemotherapy_Time",
  event_col = "Chemotherapy_Event",
  cont_col = "chemo",
  trt_col = "chemo",
  target_logHR = hr
)
chemo_cp <- result$best_cutpoint
chemo_train_hr <- result$best_logHR_train
chemo_val_hr <- result$best_logHR_val

hr_df <- rbind(hr_df, data.frame("Treatment" = c("Hormone Therapy", "Hormone Therapy", "CDK Inhibitor",
                                                 "CDK Inhibitor", "Chemotherapy", "Chemotherapy",
                                                 "Immunotherapy", "Immunotherapy"),
                                 "Dataset" = rep(c("Training", "Validation"), 2),
                                 "HR" = c(hormone_train_hr, hormone_val_hr, cdk_train_hr, cdk_val_hr,
                                          chemo_train_hr, chemo_val_hr, immune_train_hr, immune_val_hr)))

cat$`Hormone Therapy`[predictions$hormone < hormone_cp] <- "Responder"
cat$`CDK Inhibitor`[predictions$cdk < cdk_cp] <- "Responder"
cat$`Immunotherapy`[predictions$immune < immune_cp] <- "Responder"
cat$`Chemotherapy`[predictions$chemo < chemo_cp] <- "Responder"


cat$ID <- subset(ids, Dataset == "Test")$ID
library(tidyverse)

cat_long <- cat %>%
  pivot_longer(
    cols = 1:4,
    names_to = "Treatment",
    values_to = "Response"
  )

cat_long_all <- rbind(cat_long_all, cat_long)

survival$ID <- toupper(survival$ID)

outcomes <- data.frame()

for (pt in unique(cat_long$ID)){
  recommended_treatments <- subset(cat_long, Response == "Responder" & ID == pt)$Treatment
  if (length(recommended_treatments) == 0){
    out <- "No Treatments Recommended"
    missed_treatments <- NA
    given_treatments <- NA
    }
  if (length(recommended_treatments) > 0){
    treat_sub <- subset(survival, ID == pt & Medication_Combination %in% recommended_treatments)
    if (nrow(treat_sub) == 0){
      out = "Recommended Treatments Not Received"
      missed_treatments <- paste0(recommended_treatments, collapse = ", ")
      given_treatments <- NA
      }
    if (nrow(treat_sub) > 0){
      out = "Recommended Treatments Received"
      missed_treatments <- paste0(setdiff(recommended_treatments, treat_sub$Medication_Combination), collapse = ", ")
      given_treatments <- paste0(treat_sub$Medication_Combination, collapse = ", ")
      }
  }
  row <- data.frame("ID" = pt, "Outcome" = out, "Missed_Treatments" = missed_treatments, "Given_Treatments" = given_treatments)
  outcomes <- rbind(outcomes, row)
}


surv <- unique(survival[,c(1,4,5)])
surv <- subset(surv, ID %in% outcomes$ID)


outcomes <- merge(outcomes, surv, by = "ID")
outcomes <- subset(outcomes, Outcome != "No Treatments Recommended")

outcomes_all <- rbind(outcomes_all, outcomes)

###### individual medication hr plots
outcomes_ind <- data.frame()
for (pt in unique(cat_long$ID)){
  for (med in c("CDK Inhibitor", "Chemotherapy", "Hormone Therapy", "Immunotherapy")){
    recommended_treatments <- subset(cat_long, Response == "Responder" & ID == pt & Treatment == med)$Treatment
    if (length(recommended_treatments) == 0){
      out <- "Non-Responder"
      missed_treatments <- NA
      given_treatments <- NA
    }
    if (length(recommended_treatments) > 0){
      treat_sub <- subset(survival, ID == pt & Medication_Combination %in% recommended_treatments)
      if (nrow(treat_sub) == 0){
        out = "Responder: Treatment Not Received"
      }
      if (nrow(treat_sub) > 0){
        out = "Responder: Treatment Received"
      }
    }
    row <- data.frame("ID" = pt, "Outcome" = out, "Treatment" = med)
    outcomes_ind <- rbind(outcomes_ind, row)
  }
}
outcomes_ind <- merge(outcomes_ind, surv, by = "ID")
outcomes_ind_all <- rbind(outcomes_ind_all, outcomes_ind)
}

outcomes_all$Time <- outcomes_all$Time / 30.44
fit <- survfit(Surv(Time, Event) ~ Outcome, data = outcomes_all)
pal <- pal_observable("observable10", alpha = 0.9)(10)
ggsurvplot(fit, data = outcomes_all,
           xlab = "Months",
           censor.size = 2,
           ylab = "Probability of Survival",
           palette =
             c(pal[3], pal[1]),
           legend.title = "",
           conf.int = FALSE,          # Add confidence interval
           pval = TRUE,              # Add p-value
           risk.table = "nrisk_cumevents",      # Add risk table
           risk.table.col = "strata",# Risk table color by groups   # Change legend labels
           risk.table.height = 0.3, # Useful to change when you have multiple groups
           risk.table.y.text = FALSE,
           fontsize = 4)


clin <- read.csv("~//Documents//COH Breast Cohort//combined_clinical_012226.csv")
clin$ID <- toupper(clin$ID)
clin <- unique(clin[,c(2,3,9,16,4,5,6)])
clin <- subset(clin, ID %in% outcomes_all$ID)


outcomes_all <- merge(outcomes_all, clin, by = "ID")


outcomes_all$Stage[outcomes_all$Stage %in% c("Stage I", "Stage II", "Stage III")] <- "Stage I-III"

outcomes_all$Stage[outcomes_all$Stage == "Not Reported"] <- NA
outcomes_all$Stage <- factor(outcomes_all$Stage, levels = c("Stage I-III", "Stage IV"))
outcomes_all$Stage <- relevel(outcomes_all$Stage, ref = "Stage I-III")

outcomes_all$HR <- "Negative"
outcomes_all$HR[outcomes_all$ER == "Positive" | outcomes_all$PR == "Positive"] <- "Positive"

library(forestmodel)
cox <- coxph(Surv(Time, Event) ~ Outcome + Age + Stage + HR + HER2, data = outcomes_all)

fp <- forest_model(cox, covariates = c("Outcome", "Age", "Stage", "HR", "HER2"),
                   exponentiate = FALSE,
                        format_options = list(colour = "black",
                                              color = NULL,
                                              shape = 20,
                                              text_size = 4,
                                              point_size = 5,
                                              banded = FALSE
                        ))
fp

hr_plot <- ggplot(subset(hr_df, Dataset == "Training"), aes(x = Treatment, y = HR, fill = Treatment, color = Treatment)) +
  geom_boxplot(position = position_dodge(1), alpha = 0.6, outliers = F) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "lightgrey") +
  theme_classic() +
  ylab("Hazard Ratio") +
  xlab("Treatment") +
  scale_fill_observable() +
  scale_color_observable() +
  theme(plot.title = element_text(hjust = 0.5, vjust = 0.5),
        axis.title = element_text(size = 16), axis.text.y = element_text(size = 14), strip.text = element_text(size = 10),
        legend.title = element_text(size = 16), legend.text = element_text(size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 16))
hr_plot

########## hazard rations for individual medications (Supplementary Figure)
outcomes_ind_all <- merge(outcomes_ind_all, clin, by = "ID")


outcomes_ind_all$Stage[outcomes_ind_all$Stage %in% c("Stage I", "Stage II", "Stage III")] <- "Stage I-III"

outcomes_ind_all$Stage[outcomes_ind_all$Stage == "Not Reported"] <- NA
outcomes_ind_all$Stage <- factor(outcomes_ind_all$Stage, levels = c("Stage I-III", "Stage IV"))
outcomes_ind_all$Stage <- relevel(outcomes_ind_all$Stage, ref = "Stage I-III")

outcomes_ind_all$HR <- "Negative"
outcomes_ind_all$HR[outcomes_ind_all$ER == "Positive" | outcomes_ind_all$PR == "Positive"] <- "Positive"
outcomes_ind_all$Time <- outcomes_ind_all$Time / 30.44
library(forestmodel)
outcomes_ind_all$Outcome <- factor(outcomes_ind_all$Outcome, levels = c("Responder: Treatment Received",
                                                                        "Responder: Treatment Not Received",
                                                                        "Non-Responder"))

coxph_list <- list()
fp_list <- list()
plot_list = list()
for (drug in c("Hormone Therapy", "CDK Inhibitor", "Chemotherapy", "Immunotherapy")){
cox <- coxph(Surv(Time, Event) ~ Outcome + Age + Stage + HR + HER2, data = subset(outcomes_ind_all, Treatment == drug))
fit <- survfit(Surv(Time, Event) ~ Outcome, data = subset(outcomes_ind_all, Treatment == drug))
plot <- ggsurvplot(fit, data = subset(outcomes_ind_all, Treatment == drug),
                   pval = TRUE, title = drug)
plot_list[[drug]] <- plot

fp <- forest_model(cox, covariates = c("Outcome", "Age", "Stage", "HR", "HER2"),
                   exponentiate = FALSE,
                   format_options = list(colour = "black",
                                         color = NULL,
                                         shape = 20,
                                         text_size = 5,
                                         point_size = 5,
                                         banded = FALSE
                   ))+
  ggtitle(drug) +
  theme(plot.title = element_text(hjust = 0.5))
fp_list[[drug]] <- fp
}

ggarrange(plotlist = fp_list)

#### plots for individual treatments
cat_long_all <- merge(cat_long_all, outcomes_all, by = "ID")
cat_long_all <- unique(cat_long_all)

plot_list <- list()
coxph_list <- list()
for (treatment in unique(cat_long_all$Treatment)){
  sub <- subset(cat_long_all, Treatment == treatment)
  surv_sub <- subset(survival, Medication_Combination == treatment)
  sub <- subset(sub, ID %in% surv_sub$ID)
  cox <- coxph(Surv(Time, Event) ~ Response, data = sub)
  fit <- survfit(Surv(Time, Event) ~ Response, data = sub)
  plot <- ggsurvplot(fit, data = sub,
             xlab = "Days",
             censor.size = 2,
             ylab = "Probability of Survival",
             title = treatment,
             palette =
               c(pal[3], pal[1]),
             legend.title = "",
             conf.int = FALSE,          # Add confidence interval
             pval = TRUE,              # Add p-value
             risk.table = "nrisk_cumevents",      # Add risk table
             risk.table.col = "strata",# Risk table color by groups   # Change legend labels
             risk.table.height = 0.3, # Useful to change when you have multiple groups
             risk.table.y.text = FALSE,
             fontsize = 4)
  plot_list[[treatment]] <- plot

  fp <- forest_model(cox, covariates = c("Response"),
                     exponentiate = FALSE,
                     format_options = list(colour = "black",
                                           color = NULL,
                                           shape = 20,
                                           text_size = 5,
                                           point_size = 5,
                                           banded = FALSE
                     ))+
    ggtitle(treatment) +
    theme(plot.title = element_text(hjust = 0.5))
  coxph_list[[treatment]] <- fp


}

## stacked bar plots

all_props <- data.frame()
for (treatment in unique(cat_long_all$Treatment)){
  sub <- subset(cat_long_all, Treatment == treatment & Response == "Responder")
  if (nrow(sub) > 0){
  #if (treatment == "Immunotherapy"){treatment <- "Immunotherapy (CKI)"}
  surv_sub <- subset(survival, Medication_Combination == treatment)
  sub$Group <- "Not Received"
  sub$Group[sub$ID %in% surv_sub$ID] <- "Received"
  tab <- data.frame(table(sub$Group))
  tab$Treatment <- treatment
  tab$Percentage <- round((tab$Freq / sum(tab$Freq)) * 100, digits = 1)
  tab$Percentage <- paste0(tab$Percentage, " %")
  all_props <- rbind(all_props, tab)
  }
}

plot <- ggplot(all_props, aes(x = Treatment, y = Freq, fill = Var1)) +
  geom_bar(position = "fill", stat = "identity", color = NA, alpha = 0.7) +
  theme_classic() +
  ylab("Proportion of Patients") +
  #scale_fill_manual(values = c("0" = p1_light[1], "I" = p1_light[2], "II" = p1_light[3], "III" = p1_light[4], "IV" = p1_light[5],
  #                                                "Not Reported" = "white")) +
  geom_text(aes(label = Percentage), size = 5, position = position_fill(vjust = 0.5)) +
  scale_fill_manual(values = c(pal[3], pal[1]))+
  theme(axis.text = element_text(size = 12), axis.title = element_text(size = 12), legend.text = element_text(size = 12),
        legend.title = element_text(size = 12),
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))
plot



all_props_compare <- data.frame()
for (treatment in unique(cat_long_all$Treatment)){
  sub <- subset(cat_long_all, Treatment == treatment & Response == "Responder")
  if (nrow(sub) > 0){
    #if (treatment == "Immunotherapy"){treatment <- "Immunotherapy (CKI)"}
    surv_sub <- subset(survival, Medication_Combination == treatment)
    sub$Group <- "Not Received"
    sub$Group[sub$ID %in% surv_sub$ID] <- "Received"
    tab <- sub[,c(10,11,16)]
    tab$Treatment <- treatment
    all_props_compare <- rbind(all_props_compare, tab)
  }
}
all_props_compare$Stage <- as.character(all_props_compare$Stage)
all_props_compare$Stage[is.na(all_props_compare$Stage) == T] <- "Stage Unknown"
all_props_compare$Subtype[is.na(all_props_compare$Subtype) == T] <- "Subtype Unknown"

plot_list <- list()
for (treatment in unique(all_props_compare$Treatment)){
  for (group in unique(all_props_compare$Group)){
    subtype_df <- subset(all_props_compare, Treatment == treatment & Group == group)
    title = paste0(treatment, " (", group, ")")

    subtype_df2 <- rbind(data.frame(table(subset(subtype_df, Stage == "Stage I-III")$Subtype)),
                         data.frame(table(subset(subtype_df, Stage == "Stage IV")$Subtype)),
                         data.frame(table(subset(subtype_df, Stage == "Stage Unknown")$Subtype)))

    subtype_df2$Stage <- c(rep("Stage I-III", length(unique(subset(subtype_df, Stage == "Stage I-III")$Subtype))),
                           rep("Stage IV", length(unique(subset(subtype_df, Stage == "Stage IV")$Subtype))),
                           rep("Stage Unknown", length(unique(subset(subtype_df, Stage == "Stage Unknown")$Subtype))))

    colnames(subtype_df2) <- c("Subtype", "Count", "Stage")
    pal <- pal_observable("observable10")(10)

    subtype_df2$Subtype <- factor(subtype_df2$Subtype, levels = c("Subtype Unknown","HR+/HER2-", "HR+/HER2+", "HR-/HER2+", "HR-/HER2-"))
    subtype_df2$Stage <- factor(subtype_df2$Stage, levels = c("Stage Unknown", "Stage I-III", "Stage IV"))
    #subtype_df2$Count[subtype_df2$Count < 10] <- NA

    subtype_plot <- ggplot(subtype_df2, aes(x = Subtype, y = Count, fill = Stage)) +
      geom_bar(position = "fill", stat = "identity", color = NA, alpha = 0.9) +
      theme_classic() +
      ylab("Proportion of Patients") +
      ggtitle(title) +
      geom_text(aes(label = Count), size = 6, position = position_fill(vjust = 0.5)) +
      scale_fill_observable()+
      theme(axis.text = element_text(size = 16), axis.title = element_text(size = 16), legend.text = element_text(size = 14),
            legend.title = element_text(size = 14), plot.title = element_text(hjust = 0.5),
            axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))
    plot_list[[title]] <- subtype_plot
  }
}


ggarrange(plotlist = plot_list, common.legend = T, align = "hv", ncol = 2, nrow = 4)


