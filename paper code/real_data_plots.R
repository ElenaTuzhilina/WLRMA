library(dplyr)
library(ggplot2)
library(latex2exp)
library(scales)


#compare soft impute
resultsoft = readRDS("Fits/real_data_soft_info.rds")
lambdas = unique(resultsoft$parameter)
resultsoft = resultsoft %>% 
  mutate(method = factor(method, levels = c("baseline", "nesterov", "anderson")),
         parameter = factor(parameter, levels = lambdas)) %>% filter(parameter != 10 & parameter != 100)

appender = function(parameter) TeX(paste("\\lambda = $", parameter))  
resultsoft %>% 
  ggplot(aes(x = iter, y = log(delta, 10), color = method))+
  geom_hline(mapping = aes(yintercept = -8), color = "black", linetype = "dashed")+
  geom_line(linewidth = 0.4)+
  coord_cartesian(ylim = c(-8, 0))+
  facet_wrap(~parameter, scale = "free_x", ncol = 4, labeller = as_labeller(appender, default = label_parsed))+
  xlab("iteration")+
  ylab(expression(log[10](Delta)))+
  #scale_x_continuous(breaks = seq(0, 100, 5))+
  theme(legend.position = "top")
ggsave("Plots/real_data_soft_iteration.pdf", width = 7, height = 4)

resultsoft %>%
  ggplot(aes(x = time, y = log(delta, 10), color = method))+
  geom_hline(mapping = aes(yintercept = -8), color = "black", linetype = "dashed")+
  geom_line(linewidth = 0.4)+
  coord_cartesian(ylim = c(-8, 0))+
  facet_wrap(~parameter, scale = "free_x", ncol = 4, labeller = as_labeller(appender, default = label_parsed))+
  xlab("time (sec)")+
  ylab(expression(log[10](Delta)))+
  #scale_x_continuous(breaks = seq(0, 100, 5))+
  theme(legend.position = "top")
ggsave("Plots/real_data_soft_time.pdf", width = 7, height = 4)

resultsoft %>%
  ggplot(aes(x = time, y = rank, color = method))+
  geom_line(linewidth = 0.4)+
  facet_wrap(~parameter, scale = "free", ncol = 4, labeller = as_labeller(appender, default = label_parsed))+
  xlab("time (sec)")+
  ylab("rank")+
  #scale_x_continuous(breaks = seq(0, 100, 5))+
  theme(legend.position = "none")
ggsave("Plots/real_data_soft_rank.pdf", width = 7, height = 3.5)

#degrees of freedom
resultdf = readRDS("Fits/real_data_soft_dfs.rds")

resultdf %>%
  group_by(lambda, type) %>%
  summarize(rank = mean(dfs)) %>%
  mutate(method = factor(type, levels = c("A", "B"), labels = c("A fixed", "B fixed"))) %>%
  ggplot(aes(factor(lambda), rank, fill = method))+
  geom_bar(stat="identity", position = position_dodge())+
  xlab(TeX("penatly factor (\\lambda)"))+
  ylab("effective rank")+
  scale_y_continuous(breaks = seq(0, 60, 10))
ggsave("Plots/real_data_df_AB.pdf", width = 6, height = 3)

resultdf %>%
  group_by(lambda, type) %>%
  summarize(rank = round(mean(dfs)))

#compare hard
resulthard = readRDS("Fits/real_data_hard_info.rds")
ks = unique(resulthard$parameter)
resulthard = resulthard %>% 
  mutate(method = factor(method, levels = c("baseline", "nesterov", "anderson")),
         parameter = factor(parameter, levels = ks))

appender = function(parameter) TeX(paste("k = $", parameter))  
resulthard %>%
  ggplot(aes(x = time, y = log(delta, 10), color = method))+
  geom_hline(mapping = aes(yintercept = -8), color = "black", linetype = "dashed")+
  geom_line(linewidth = 0.4)+
  coord_cartesian(ylim = c(-8, 0))+
  facet_wrap(~parameter, scale = "free_x", ncol = 4, labeller = as_labeller(appender, default = label_parsed), )+
  xlab("time (sec)")+
  ylab(expression(log[10](Delta)))+
  #scale_x_continuous(breaks = seq(0, 100, 5))+
  theme(legend.position = "top")
ggsave("Plots/real_data_hard_time.pdf", width = 7, height = 4.5)

resultsoft = readRDS("Fits/real_data_soft_info.rds") %>% filter(method == "anderson" & parameter %in% c(20,30,40,50))
resulthard = readRDS("Fits/real_data_hard_info.rds") %>% filter(method == "anderson") %>%
  mutate(parameter = case_when(parameter == 3 ~ 50,
                               parameter == 5 ~ 40, 
                               parameter == 12 ~ 30, 
                               parameter == 29 ~ 20))
rbind(resultsoft, resulthard) %>%
  ggplot(aes(x = time, y = log(loss_no_penalty), color = type))+
  geom_line(linewidth = 0.4)+
  facet_wrap(~parameter, scale = "free_x", ncol = 4, labeller = as_labeller(appender, default = label_parsed))+
  xlab("time (sec)")+
  ylab(TeX("$\\|\\sqrt{W} * (X - AB^T)\\|^2_F$"))+
  #scale_x_continuous(breaks = seq(0, 100, 5))+
  theme(legend.position = "top")

