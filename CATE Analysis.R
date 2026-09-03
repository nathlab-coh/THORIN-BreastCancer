
# CATE Analysis to compare treatment effects across subgroups (Figure 4f, Supplementary Figure)

#install.packages("precmed")
library(precmed)
combined <- read.csv("~//Documents//COH Breast Cohort//DRE//combined_data_treatments_all.csv")

data_all <- data.frame()
all_proj <- data.frame()
for (fold in 1:5){
  
  #########################################################
    ##### Load Survival Data and THORIN Predictions ######
  #########################################################
  
  
  predictions_test <- read.csv(paste0("multitask_model_test_outcomes_fold", fold, "_FINAL.csv"))
  predictions <- read.csv(paste0("multitask_model_train_outcomes_fold", fold, "_FINAL.csv"))

  ids <- read.csv(paste0("~//Final Datasets//fold", fold, "_ids.csv"))

  train_y <- read.csv(paste0("~//Final Datasets//survival_group_fold", fold, "_train.csv"))
  val_y <- read.csv(paste0("~//Final Datasets//survival_group_fold", fold, "_val.csv"))
  test_y <- read.csv(paste0("~//Final Datasets//survival_group_fold", fold, "_test.csv"))


  survival <- read.csv("patient_treatment_responses_time_to_change.csv")
  survival$Medication_Combination[grep("Hormone", survival$Medication_Combination)] <- "Hormone Therapy"
  survival$Medication_Combination[grep("Chemotherapy", survival$Medication_Combination)] <- "Chemotherapy"
  survival$Medication_Combination[survival$Medication_Combination == "Immunotherapy (CKI)"] <- "Immunotherapy"
  
  cat <- data.frame(matrix(NA, ncol = ncol(predictions), nrow = nrow(predictions)))
  colnames(cat) <- c("Hormone Therapy", "Immunotherapy",
                     "CDK Inhibitor", "Chemotherapy")
  cat$`Hormone Therapy` <- predictions$hormone
  cat$`CDK Inhibitor` <- predictions$cdk
  cat$Chemotherapy <- predictions$chemo
  cat$Immunotherapy <- predictions$immune
  
  library(survcomp)
  train_predictions <- cbind(predictions, train_y)
  
  cat$ID <- subset(ids, Dataset == "Train")$ID
  predictions$ID <- subset(ids, Dataset == "Train")$ID
  
  cat_test <- data.frame(matrix(NA, ncol = ncol(predictions_test), nrow = nrow(predictions_test)))
  colnames(cat_test) <- c("Hormone Therapy", "Immunotherapy",
                     "CDK Inhibitor", "Chemotherapy")
  
  cat_test$`Hormone Therapy` <- predictions_test$hormone
  cat_test$`CDK Inhibitor` <- predictions_test$cdk
  cat_test$Chemotherapy <- predictions_test$chemo
  cat_test$Immunotherapy <- predictions_test$immune
  
  cat_test$ID <- subset(ids, Dataset == "Test")$ID
  predictions_test$ID <- subset(ids, Dataset == "Test")$ID
  
  #########################################################
##### Run CATE for Overall Treatment Effects each Drug ######
  #########################################################
  
  library(grf)
  library(survival)
  library(dplyr)
  library(survminer)
  meds <- c("Hormone Therapy", "Immunotherapy",
            "CDK Inhibitor", "Chemotherapy")
  i <- 2
  plots <- list()
  
  for (med in meds){
  surv_sub <- survival
  surv_sub$ID <- toupper(surv_sub$ID)
  pred_sub <- subset(cat, ID %in% surv_sub$ID)
  clin <- read.csv("combined_clinical.csv")
  clin$ID <- toupper(clin$ID)
  clin <- unique(clin[,c(2,3,9,16,4,5,6)])
  clin <- subset(clin, ID %in% pred_sub$ID)
  pred_sub <- merge(pred_sub, clin, by = "ID")
  pred_sub$Stage[pred_sub$Stage %in% c("Stage I", "Stage II", "Stage III")] <- "Stage I-III"
  pred_sub$Stage <- factor(pred_sub$Stage, levels = c("Stage I-III", "Stage IV", "Not Reported"))
  data <- unique(merge(pred_sub, surv_sub, by = "ID"))
  on_med <- unique(subset(data, Medication_Combination == med)$ID)
  data <- unique(data[,-c(13,16)])
  data$trt <- 0
  data$trt[data$ID %in% on_med] <- 1
  data$trt <- factor(data$trt, levels = c(1,0))
  
  ##
  pred_sub_test <- subset(cat_test, ID %in% surv_sub$ID)
  clin <- read.csv("combined_clinical.csv")
  clin$ID <- toupper(clin$ID)
  clin <- unique(clin[,c(2,3,9,16,4,5,6)])
  clin <- subset(clin, ID %in% pred_sub_test$ID)
  pred_sub_test <- merge(pred_sub_test, clin, by = "ID")
  pred_sub_test$Stage[pred_sub_test$Stage %in% c("Stage I", "Stage II", "Stage III")] <- "Stage I-III"
  pred_sub_test$Stage <- factor(pred_sub_test$Stage, levels = c("Stage I-III", "Stage IV", "Not Reported"))
  data_test <- merge(pred_sub_test, surv_sub, by = "ID")
  on_med <- unique(subset(data_test, Medication_Combination == med)$ID)
  data_test <- unique(data_test[,-c(13,16)])
  data_test$trt <- 0
  data_test$trt[data_test$ID %in% on_med] <- 1
  data_test$trt <- factor(data_test$trt, levels = c(1,0))
  
  data$HR <- "Negative"
  data$HR[data$ER == "Positive" | data$PR == "Positive"] <- "Positive"
  x <- data.matrix(data[,c("Age", "Stage", "HR", "HER2")])
  
  csf <- causal_survival_forest(
    X = x,
    Y = as.numeric(data$Time),
    W = as.numeric(data$trt),
    D = as.numeric(data$Event),
    horizon = quantile(subset(data, Event == 1)$Time, 0.8),
    num.trees = 2000,
    seed = 1
  )
  
  cate_pred <- predict(csf)$predictions
  
  data_test$HR <- "Negative"
  data_test$HR[data_test$ER == "Positive" | data_test$PR == "Positive"] <- "Positive"
  x_test <- data.matrix(data_test[,c("Age", "Stage", "HR", "HER2")])
  test_pred <- predict(csf, x_test)$predictions
  data_test$CATE <- as.numeric(test_pred)
  
  data$CATE <- cate_pred
  colnames(data)[i] <- "Response"
  colnames(data_test)[i] <- "Response"
  i <- i+1
  #Negative CATE → treatment reduces hazard (benefit)
  #Positive CATE → treatment increases hazard (harm)
  data_sub <- subset(data, trt == 1)
  data_sub$Treatment <- med
  
  data_sub_test <- subset(data_test, trt == 1)
  data_sub_test$Treatment <- med
  
  proj <- best_linear_projection(csf, x)
  proj <- data.frame("Estimate" = proj[6], "p.value" = proj[24],
                     "Treatment" = med)
  
  data_sub_test$Fold <- paste("Fold", fold)
  
  
  # Precompute scaling factors safely
  pos_vals <- data_sub$CATE[data_sub$CATE > 0]
  neg_vals <- data_sub$CATE[data_sub$CATE < 0]
  
  pos_max <- if (length(pos_vals) > 0) max(pos_vals, na.rm = TRUE) else NA_real_
  neg_min <- if (length(neg_vals) > 0) min(neg_vals, na.rm = TRUE) else NA_real_
  
  
  resp_min <- min(data_sub_test$Response, na.rm = TRUE)
  resp_max <- max(data_sub_test$Response, na.rm = TRUE)
  resp_range <- resp_max - resp_min
  
  
  #########################################################
    ##### Split Patients into CATE response groups ######
  #########################################################
  
  data_sub_test <- data_sub_test %>%
    mutate(
      CATE_group = ntile(CATE, 3),
      CATE_group = case_when(
        CATE_group == 1 ~ "Low (Best Effect)",
        CATE_group == 2 ~ "Medium (Moderate Effect)",
        CATE_group == 3 ~ "High (Worst Effect)"
      )
    )
  
  
  data_sub_test <- data_sub_test %>%
    mutate(
      Response = (Response - min(Response, na.rm = TRUE)) /
        (max(Response, na.rm = TRUE) - min(Response, na.rm = TRUE))
    )
  
  
  
  
  data_all <- rbind(data_all, data_sub_test[,c("CATE", "CATE_group", "Response", "Treatment", "Fold")])
  
  }
}

pal <- pal_observable("observable10", alpha = 1)(10)

#########################################################
##### Plot Final CATE Scores vs THORIN Predictions ######
#########################################################

data_all$CATE_group <- factor(data_all$CATE_group, levels = c("Low (Best Effect)", "Medium (Moderate Effect)", "High (Worst Effect)"))


cate_plot <- ggplot(data_all, aes(x = CATE_group, fill = CATE_group, y = Response, color = CATE_group)) +
  geom_boxplot(position = position_dodge(1), alpha = 0.6) +
  theme_classic() +
  ylab("Predicted Response") +
  xlab("CATE Group") +
  scale_fill_observable() +
  scale_color_observable() +
  theme(plot.title = element_text(hjust = 0.5, vjust = 0.5),
        axis.title = element_text(size = 16), axis.text.y = element_text(size = 14), strip.text = element_text(size = 10),
        legend.title = element_text(size = 16), legend.text = element_text(size = 14),
        axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
  facet_wrap(~Treatment, nrow = 1)+
  stat_compare_means(comparisons = list(c("Low (Best Effect)", "Medium (Moderate Effect)"), c("Low (Best Effect)", "High (Worst Effect)"), c("Medium (Moderate Effect)", "High (Worst Effect)")))


cate_plot

cate_overall_plot <- ggplot(data_all, aes(x = Treatment, fill = Treatment, y = CATE, color = Treatment)) +
  geom_boxplot(position = position_dodge(1), alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "lightgrey") +
  theme_classic() +
  ylab("CATE") +
  xlab("Treatment") +
  scale_fill_observable() +
  scale_color_observable() +
  theme(plot.title = element_text(hjust = 0.5, vjust = 0.5),
        axis.title = element_text(size = 16), axis.text.y = element_text(size = 14), strip.text = element_text(size = 10),
        legend.title = element_text(size = 16), legend.text = element_text(size = 14),
        axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 16)) +
  stat_compare_means()


cate_overall_plot


