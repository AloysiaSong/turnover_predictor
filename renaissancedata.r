# =============================================================================
# 面板数据结构重构与标准化变量构造
# =============================================================================

library(tidyverse)
library(psych)      # 因子分析
library(lavaan)     # 结构方程模型/CFA
library(lsa)        # 余弦相似度

# -----------------------------------------------------------------------------
# 1. 读取数据
# -----------------------------------------------------------------------------
df_raw <- read.csv("/Users/yu/code/code2510/laborecon/final_data_merged_complete.csv", stringsAsFactors = FALSE)

# -----------------------------------------------------------------------------
# 2. 创建个体与企业标识符
# -----------------------------------------------------------------------------
# 生成唯一个体ID
df_raw$person_id <- 1:nrow(df_raw)

# 基于企业类型、规模、城市创建企业分组ID（近似firm_id）
df_raw <- df_raw %>%
  mutate(
    firm_group = paste(company_type, company_size, city, sep = "_"),
    firm_id = as.integer(factor(firm_group))
  )

# -----------------------------------------------------------------------------
# 3. 变量映射与标准化命名
# -----------------------------------------------------------------------------
df_base <- df_raw %>%
  mutate(
    # 时间标识
    wave = "t0",
    t_months = 0,
    
    # 企业侧暴露变量
    firm_type = company_type,
    firm_size = company_size,
    firm_size_code = case_when(
      grepl("100以下|<100", company_size, ignore.case = TRUE) ~ 1,
      grepl("100-999|100–999", company_size, ignore.case = TRUE) ~ 2,
      grepl("1000-4999|1000–4999", company_size, ignore.case = TRUE) ~ 3,
      grepl("5000以上|5000\\+|>=5000", company_size, ignore.case = TRUE) ~ 4,
      TRUE ~ NA_real_
    ),
    # 对数规模（使用区间中点）
    ln_size = case_when(
      firm_size_code == 1 ~ log(50),
      firm_size_code == 2 ~ log(550),
      firm_size_code == 3 ~ log(2750),
      firm_size_code == 4 ~ log(7000),
      TRUE ~ NA_real_
    ),
    
    # 个体控制变量
    tenure_total = total_experience,
    tenure_current = tenure_current,
    last_job_change = time_since_last_change,
    training_hours_12m = training_hours,
    commute_minutes = commute_time,
    city_satisfaction = Q11,
    
    # 薪酬变量
    salary_band = Q20,
    
    # 离职结果变量
    turnover_3m = job_change_3m,
    turnover_pref_6_12m = job_change_prob / 10  # 标准化到0-1
  )

# -----------------------------------------------------------------------------
# 4. 岗位类别多热编码（Q5系列）
# -----------------------------------------------------------------------------
function_cols <- paste0("Q5_", 1:13)
function_names <- c("data", "algorithm", "analysis", "product", "operation",
                    "sales", "hr", "finance", "legal", "admin", "rd", 
                    "production", "other")

for (i in 1:13) {
  col_name <- paste0("position_", function_names[i])
  df_base[[col_name]] <- df_base[[function_cols[i]]]
}

# -----------------------------------------------------------------------------
# 5. P-J Fit 构造（两层结构）
# -----------------------------------------------------------------------------
# Q12系列作为P-J Fit指标（假设Q12_1-Q12_5）
fit_items <- paste0("Q12_", 1:5)

# 检查Q12数据是否存在
if (all(fit_items %in% colnames(df_base))) {
  # 标准化
  df_fit <- df_base %>%
    select(all_of(fit_items)) %>%
    mutate(across(everything(), ~scale(.)[,1]))
  
  # 探索性因子分析（EFA）
  fa_result <- fa(df_fit, nfactors = 2, rotate = "varimax", fm = "ml")
  
  # 保存因子得分
  df_base$fit_skill_latent <- fa_result$scores[, 1]
  df_base$fit_psy_latent <- fa_result$scores[, 2]
  
  # 简单均分作为备选
  df_base$fit_skill_mean <- rowMeans(df_base[, fit_items], na.rm = TRUE)
  
} else {
  warning("Q12系列变量不完整，跳过P-J Fit构造")
}

# -----------------------------------------------------------------------------
# 6. 技能结构匹配
# -----------------------------------------------------------------------------
# Q13为使用频率，Q14为熟练度（假设各有15项）
skill_freq_cols <- paste0("Q13_", 1:15)
skill_prof_cols <- paste0("Q14_", 1:15)

if (all(c(skill_freq_cols, skill_prof_cols) %in% colnames(df_base))) {
  # 提取技能矩阵
  skill_freq_mat <- as.matrix(df_base[, skill_freq_cols])
  skill_prof_mat <- as.matrix(df_base[, skill_prof_cols])
  
  # (A) 差值法：技能缺口
  skill_gap_mat <- skill_prof_mat - skill_freq_mat
  
  # 聚合为平均缺口
  df_base$skill_gap_mean <- rowMeans(abs(skill_gap_mat), na.rm = TRUE)
  
  # 转换为匹配度：1 - 标准化缺口
  df_base$fit_skill_struct_gap <- 1 - (df_base$skill_gap_mean / 4)
  
  # (B) 余弦相似度法
  df_base$fit_skill_struct_cos <- sapply(1:nrow(df_base), function(i) {
    freq_vec <- skill_freq_mat[i, ]
    prof_vec <- skill_prof_mat[i, ]
    
    # 处理缺失值
    valid_idx <- !is.na(freq_vec) & !is.na(prof_vec)
    if (sum(valid_idx) < 2) return(NA)
    
    # 计算余弦相似度
    cosine(freq_vec[valid_idx], prof_vec[valid_idx])
  })
  
  # 保存各维度技能缺口
  for (i in 1:15) {
    df_base[[paste0("skill_gap_", i)]] <- skill_gap_mat[, i]
  }
  
} else {
  warning("技能变量不完整，跳过技能匹配构造")
}

# -----------------------------------------------------------------------------
# 7. 经济损失感知（两层）
# -----------------------------------------------------------------------------
# Q16系列（假设Q16_1-Q16_5）
econloss_items <- paste0("Q16_", 1:5)

if (all(econloss_items %in% colnames(df_base))) {
  # 假设：前3项为薪酬相关，后2项为心理相关
  econloss_salary_items <- paste0("Q16_", 1:3)
  econloss_psy_items <- paste0("Q16_", 4:5)
  
  # 标准化并聚合
  df_base$econloss_salary <- rowMeans(
    scale(df_base[, econloss_salary_items]), 
    na.rm = TRUE
  )
  
  df_base$econloss_psy <- rowMeans(
    scale(df_base[, econloss_psy_items]), 
    na.rm = TRUE
  )
  
  # 整体经济损失（所有项均分）
  df_base$econloss_total <- rowMeans(
    df_base[, econloss_items], 
    na.rm = TRUE
  )
  
  # 二因子模型（CFA）- 可选
  cfa_model <- '
    econloss_salary =~ Q16_1 + Q16_2 + Q16_3
    econloss_psy =~ Q16_4 + Q16_5
  '
  
  tryCatch({
    cfa_fit <- cfa(cfa_model, data = df_base)
    
    # 提取因子得分
    factor_scores <- lavPredict(cfa_fit)
    df_base$econloss_salary_latent <- factor_scores[, "econloss_salary"]
    df_base$econloss_psy_latent <- factor_scores[, "econloss_psy"]
    
    # 保存模型拟合指数
    fit_indices <- fitMeasures(cfa_fit, c("cfi", "tli", "rmsea", "srmr"))
    print("经济损失CFA拟合指数：")
    print(fit_indices)
    
  }, error = function(e) {
    warning("CFA拟合失败，使用简单聚合")
  })
  
} else {
  warning("Q16系列变量不完整，跳过经济损失构造")
}

# -----------------------------------------------------------------------------
# 8. 构造长格式面板数据框架
# -----------------------------------------------------------------------------
# 选择核心变量
panel_vars <- c(
  # 标识符
  "person_id", "firm_id", "wave", "t_months",
  
  # 企业暴露变量
  "firm_type", "firm_size", "firm_size_code", "ln_size",
  
  # 地理
  "province", "city", "province_code",
  
  # 个体控制
  "tenure_total", "tenure_current", "last_job_change",
  "training_hours_12m", "commute_minutes", "city_satisfaction",
  
  # 薪酬
  "salary_band",
  
  # P-J Fit
  grep("^fit_", names(df_base), value = TRUE),
  
  # 技能匹配
  grep("^skill_(gap|freq|prof)", names(df_base), value = TRUE),
  
  # 经济损失
  grep("^econloss_", names(df_base), value = TRUE),
  
  # 岗位类别
  grep("^position_", names(df_base), value = TRUE),
  
  # 离职结果
  "turnover_3m", "turnover_pref_6_12m"
)

# 保留存在的变量
panel_vars <- panel_vars[panel_vars %in% names(df_base)]

# 创建长格式数据（当前仅t0）
panel_long <- df_base %>%
  select(all_of(panel_vars)) %>%
  arrange(person_id, wave)

# -----------------------------------------------------------------------------
# 9. 生成宽格式（用于某些分析）
# -----------------------------------------------------------------------------
# 当有多个时间点时，可以转为宽格式
# panel_wide <- panel_long %>%
#   pivot_wider(
#     id_cols = c(person_id, firm_id),
#     names_from = wave,
#     values_from = -c(person_id, firm_id, wave, t_months)
#   )

# -----------------------------------------------------------------------------
# 10. 描述性统计与数据质量检查
# -----------------------------------------------------------------------------
# 核心变量描述统计
key_vars <- c("tenure_total", "tenure_current", "training_hours_12m",
              "commute_minutes", "city_satisfaction", "salary_band",
              "turnover_3m", "turnover_pref_6_12m")

desc_stats <- panel_long %>%
  select(all_of(key_vars[key_vars %in% names(panel_long)])) %>%
  summary()

print("=== 核心变量描述统计 ===")
print(desc_stats)

# 企业分布
print("\n=== 企业类型分布 ===")
print(table(panel_long$firm_type))

print("\n=== 企业规模分布 ===")
print(table(panel_long$firm_size))

# 缺失值检查
missing_pct <- colMeans(is.na(panel_long)) * 100
print("\n=== 缺失值比例（>5%的变量）===")
print(missing_pct[missing_pct > 5])

# -----------------------------------------------------------------------------
# 11. 保存数据
# -----------------------------------------------------------------------------
# 保存长格式面板数据
write.csv(panel_long, "panel_data_long.csv", row.names = FALSE, 
          fileEncoding = "UTF-8")

# 保存带所有中间变量的完整数据
write.csv(df_base, "panel_data_full.csv", row.names = FALSE,
          fileEncoding = "UTF-8")

# 保存数据字典
var_dict <- data.frame(
  variable = names(panel_long),
  type = sapply(panel_long, class),
  n_missing = colSums(is.na(panel_long)),
  pct_missing = round(colMeans(is.na(panel_long)) * 100, 2)
)

write.csv(var_dict, "data_dictionary.csv", row.names = FALSE,
          fileEncoding = "UTF-8")

print("\n=== 数据处理完成 ===")
print(paste("面板长格式数据：", nrow(panel_long), "行 ×", 
            ncol(panel_long), "列"))
print("已保存：")
print("  - panel_data_long.csv（核心面板数据）")
print("  - panel_data_full.csv（含所有中间变量）")
print("  - data_dictionary.csv（数据字典）")

# -----------------------------------------------------------------------------
# 12. 为未来随访准备数据模板
# -----------------------------------------------------------------------------
# 创建t1和t2的空模板（用于后续随访数据录入）
panel_template_t1 <- panel_long %>%
  select(person_id, firm_id) %>%
  mutate(
    wave = "t1",
    t_months = 3,
    # 其他变量留空，待填充
    turnover_3m = NA,
    turnover_pref_6_12m = NA
  )

panel_template_t2 <- panel_long %>%
  select(person_id, firm_id) %>%
  mutate(
    wave = "t2",
    t_months = 12,
    turnover_3m = NA,
    turnover_pref_6_12m = NA
  )

# 保存随访模板
write.csv(panel_template_t1, "followup_template_t1.csv", 
          row.names = FALSE, fileEncoding = "UTF-8")
write.csv(panel_template_t2, "followup_template_t2.csv", 
          row.names = FALSE, fileEncoding = "UTF-8")

print("\n已生成随访数据模板：")
print("  - followup_template_t1.csv（3个月随访）")
print("  - followup_template_t2.csv（6-12个月随访）")