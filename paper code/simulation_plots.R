library(dplyr)
library(ggplot2)
library(latex2exp)
library(scales)

#compare hard impute
resulthard = readRDS("Fits/simulation_hard.rds") %>% 
  filter(iter <= 200) %>%
  mutate(method = factor(method, levels = c("baseline", "nesterov", "anderson")), 
         parameter = paste("k =", parameter))

resulthard %>%
  ggplot(aes(x = iter, y = log(delta, 10), color = method))+
  geom_hline(mapping = aes(yintercept = -8), color = "black", linetype = "dashed")+
  geom_line(linewidth = 0.4)+
  coord_cartesian(ylim = c(-8, 0))+
  facet_grid(~parameter)+
  xlab("iteration (t)")+
  ylab(expression(log[10](Delta)))+
  #scale_x_continuous(breaks = seq(0, 300, 50))+
  theme(legend.position = "top")+
  labs(color = "acceleration")
ggsave("Plots/simulation_hard_delta.pdf", width = 7, height = 3)

resulthard %>%
  group_by(parameter) %>%
  mutate(loss = loss - min(loss)) %>%
  filter(loss > 0) %>%
  ggplot(aes(x = iter, y = log(loss, 10), color = method))+
  geom_line(linewidth = 0.4)+
  facet_grid(~parameter)+
  xlab("iteration (t)")+
  ylab(expression(log[10](loss(X) - loss('X*'))))+
  #scale_x_continuous(breaks = seq(0, 300, 10))+
  theme(legend.position = "none")+
  scale_y_continuous(breaks = seq(0, -10, -2))+
  labs(color = "acceleration")
ggsave("Plots/simulation_hard_loss.pdf", width = 7, height = 2.6)

#compare soft impute
resultsoft = readRDS("Fits/simulation_soft.rds")
lambdas = unique(resultsoft$parameter)
resultsoft = resultsoft %>% 
  mutate(method = factor(method, levels = c("baseline", "nesterov", "anderson")),
         parameter = factor(parameter, levels = lambdas))
appender = function(parameter) TeX(paste("\\lambda = $", parameter))  
  
resultsoft %>%
  ggplot(aes(x = iter, y = log(delta, 10), color = method))+
  geom_hline(mapping = aes(yintercept = -8), color = "black", linetype = "dashed")+
  geom_line(linewidth = 0.4)+
  coord_cartesian(ylim = c(-8, 0))+
  facet_grid(~parameter, scale = "free", labeller = as_labeller(appender, default = label_parsed))+
  xlab("iteration (t)")+
  ylab(expression(log[10](Delta)))+
  #scale_x_continuous(breaks = seq(0, 100, 5))+
  theme(legend.position = "none")
ggsave("Plots/simulation_soft_delta.pdf", width = 7, height = 2.6)

resultsoft %>%
  group_by(parameter) %>%
  mutate(loss = loss - min(loss)) %>%
  filter(loss > 0) %>%
  ggplot(aes(x = iter, y = log(loss, 10), color = method))+
  geom_line(linewidth = 0.4)+
  facet_grid(~parameter, scale = "free", labeller = as_labeller(appender, default = label_parsed))+
  xlab("iteration (t)")+
  ylab(expression(log[10](loss(X) - loss('X*'))))+
  #scale_x_continuous(breaks = seq(0, 300, 10))+
  theme(legend.position = "none")+
  scale_y_continuous(breaks = seq(0, -10, -2))+
  labs(color = "acceleration")
ggsave("Plots/simulation_soft_loss.pdf", width = 7, height = 2.6)

#different initialization
resultinit = readRDS("Fits/simulation_hard_init.rds")
resultinit %>% group_by(parameter) %>%
  filter(iter <= 200) %>%
  mutate(method = factor(method, levels = c("baseline", "nesterov", "anderson")),
          init = factor(init, levels = c("zero", "random", "low-rank", "warm")), 
         parameter = paste("k =", parameter)) %>%
  ggplot(aes(iter, log(delta, 10), color = init, group = init))+
  geom_line(linewidth = 0.4)+
  facet_grid(method~parameter)+
  coord_cartesian(ylim = c(-8, 0))+
  ylab(expression(log[10](Delta)))+
  xlab("iteration (t)")+
  theme(legend.position = "top")+
  labs(color = "initialization")+
  geom_hline(mapping = aes(yintercept = -8), color = "black", linetype = "dashed")
ggsave("Plots/simulation_hard_init_delta.pdf", width = 7, height = 6)

resultinit %>% group_by(parameter) %>%
  mutate(loss = loss - min(loss),
         init = factor(init, levels = c("zero", "random", "low-rank", "warm"))) %>%
  filter(iter <= 200, loss > 0) %>%
  mutate(method = factor(method, levels = c("baseline", "nesterov", "anderson")), 
         parameter = paste("k =", parameter)) %>%
  ggplot(aes(iter, log(loss, 10), color = init, group = init))+
  geom_line(linewidth = 0.4)+
  facet_grid(method~parameter)+
  ylab(expression(log[10](loss(X) - loss('X*'))))+
  xlab("iteration (t)")+
  scale_y_continuous(breaks = seq(0, -10, -2))+
  theme(legend.position = "top")+
  labs(color = "initialization")
ggsave("Plots/simulation_hard_init_loss.pdf", width = 7, height = 6)

#different SNR and rank
resultrk = readRDS("Fits/simulation_soft_sigma_rank.rds")
lambdas = unique(resultrk$parameter)
appender = function(parameter) TeX(paste("\\lambda = $", parameter))  
resultrk %>% filter(sigma == 1) %>%
  mutate(method = factor(method, levels = c("baseline", "nesterov", "anderson")), 
         r = as.factor(r),
         parameter = factor(parameter, levels = lambdas)) %>%
  ggplot(aes(iter, log(delta, 10), color = r))+
  geom_line(linewidth = 0.4)+
  facet_grid(method~parameter,  scales = "free", labeller = labeller(parameter = as_labeller(appender, default = label_parsed)))+
  coord_cartesian(ylim = c(-8, 0))+
  geom_hline(mapping = aes(yintercept = -8), color = "black", linetype = "dashed")+
  ylab(expression(log[10](Delta)))+
  xlab("iteration (t)")+
  theme(legend.position = "top")+
  labs(color = "r")
ggsave("Plots/simulation_soft_rank.pdf", width = 7, height = 6)

resultrk %>% filter(r == 75) %>%
  mutate(method = factor(method, levels = c("baseline", "nesterov", "anderson")), 
         sigma = as.factor(sigma),
         parameter = factor(parameter, levels = lambdas)) %>%
  ggplot(aes(iter, log(delta, 10), color = sigma))+
  geom_line(linewidth = 0.4)+
  facet_grid(method~parameter,  scales = "free", labeller = labeller(parameter = as_labeller(appender, default = label_parsed)))+
  coord_cartesian(ylim = c(-8, 0))+
  ylab(expression(log[10](Delta)))+
  xlab("iteration (t)")+
  theme(legend.position = "top")+
  labs(color = expression(sigma))
ggsave("Plots/simulation_soft_sigma.pdf", width = 7, height = 6)

#different depth
resultaa = readRDS("Fits/simulation_soft_depth.rds")
lambdas = unique(resultaa$parameter)
appender = function(parameter) TeX(paste("\\lambda = $", parameter))  
resultaa %>% 
  mutate(parameter = factor(parameter, levels = lambdas),
         depth = as.factor(depth)) %>%
  ggplot(aes(x = iter, y = log(delta, 10), color = depth))+
  geom_hline(mapping = aes(yintercept = -8), color = "black", linetype = "dashed")+
  geom_line(linewidth = 0.4)+
  coord_cartesian(ylim = c(-8, 0))+
  facet_grid(~parameter,  scales = "free", labeller = labeller(parameter = as_labeller(appender, default = label_parsed)))+
  xlab("iteration (t)")+
  ylab(expression(log[10](Delta)))+
  scale_x_continuous(breaks = seq(0, 40, 5))+
  theme(legend.position = "top")+
  labs(color = "m")
ggsave("Plots/simulation_soft_depth_delta.pdf", width = 7, height = 3)

resultaa %>% group_by(parameter) %>%
  mutate(loss = loss - min(loss)) %>%
  mutate(parameter = factor(parameter, levels = lambdas),
         depth = as.factor(depth)) %>%
  filter(loss > 0) %>%
ggplot(aes(x = iter, y = log(loss, 10), color = depth))+
  geom_line(linewidth = 0.4)+
  facet_grid(~parameter,  scales = "free", labeller = labeller(parameter = as_labeller(appender, default = label_parsed)))+
  xlab("iteration (t)")+
  ylab(expression(log[10](loss(X) - loss('X*'))))+
  scale_x_continuous(breaks = seq(0, 30, 5))+
  theme(legend.position = "none")+
  #scale_y_continuous(breaks = seq(0, -10, -2))+
  labs(color = "m")
ggsave("Plots/simulation_soft_depth_loss.pdf", width = 7, height = 2.6)


#different reg hard
resultraa = readRDS("Fits/simulation_hard_reg.rds") %>% filter(!gamma %in% c(0.0001), reg_depth == 3)
ds = unique(resultraa$reg_depth)

resultraa %>% group_by(reg_depth) %>% 
  mutate(loss = loss - min(loss)) %>% 
  mutate(reg_depth = factor(paste("d =", reg_depth), levels = paste("d =", ds)), gamma = as.factor(gamma)) %>%
  filter(loss > 0) %>%
  ggplot(aes(x = iter, y = log(loss, 10), color = gamma))+
  geom_line(linewidth = 0.4)+
  #facet_wrap(~reg_depth)+
  xlab("iteration (t)")+
  ylab(expression(log[10](loss(X) - loss('X*'))))+
  #scale_x_continuous(breaks = seq(0, 300, 10))+
  #theme(legend.position = "none")+
  scale_y_continuous(breaks = seq(0, -10, -2))+
  labs(color = expression(gamma))
ggsave("Plots/simulation_hard_reg_loss.pdf", width = 3.7, height = 3)

resultraa %>% 
  mutate(reg_depth = factor(paste("d =", reg_depth), levels = paste("d =", ds)), gamma = as.factor(gamma)) %>%
  ggplot(aes(x = iter, y = log(delta, 10), color = gamma))+
  geom_hline(mapping = aes(yintercept = -8), color = "black", linetype = "dashed")+
  geom_line(linewidth = 0.4)+
  coord_cartesian(ylim = c(-8, 0))+
  #facet_grid(~reg_depth, scale = "free")+
  xlab("iteration (t)")+
  ylab(expression(log[10](Delta)))+
  #scale_x_continuous(breaks = seq(0, 300, 10))+
  #theme(legend.position = "none")+
  labs(color = expression(gamma))
ggsave("Plots/simulation_hard_reg_delta.pdf", width = 6, height = 3)


coefraa = readRDS("Fits/simulation_hard_reg_coef.rds") %>% filter(gamma != 0.0001)
appender = function(parameter) TeX(paste("\\gamma = $", parameter))  
coefraa %>% 
  mutate(gamma = as.factor(gamma)) %>%
  ggplot(aes(x = iter, y = value, color = coef))+
  geom_line(linewidth = 0.3)+
  facet_wrap(~gamma, scale = "free_y", nrow = 2, labeller = labeller(gamma = as_labeller(appender, default = label_parsed)))+
  xlab("iteration (t)")+
  ylab(TeX("Anderson coefficients (\\alpha)"))+
  scale_x_continuous(breaks = seq(0, 200, 50))+
  theme(legend.position = "none")+
  scale_color_brewer(palette = "Set1")
ggsave("Plots/simulation_hard_reg_coef.pdf", width = 7, height = 4)


#degrees-of-freedom definition
resultdf = readRDS("Fits/simulation_df.rds")
resultdf$info %>% filter(method %in% c("soft", "hard"))

resultdf$info %>%
  filter(method %in% c("hard with df (A fixed)", "hard with df (B fixed)")) %>%
  mutate(method = factor(method, levels = c("hard with df (A fixed)", "hard with df (B fixed)"), labels = c("A fixed", "B fixed"))) %>%
  ggplot(aes(factor(lambda), rank, fill = method))+
  geom_bar(stat="identity", position=position_dodge())+
  #geom_errorbar(aes(ymin = rank_min, ymax = rank_max), width=.2,
                #position = position_dodge(1))+
  xlab(TeX("penatly factor (\\lambda)"))+
  ylab("effective rank (er)")+
  scale_y_continuous(breaks = seq(0, 60, 10))+
  labs(fill = "")
ggsave("Plots/simulation_df_AB.pdf", width = 6, height = 3)

resultdfinfo = resultdf$info %>% filter(method != "hard with df (B fixed)") %>%
  mutate(method = ifelse(method == "hard with df (A fixed)", "hard with effective rank", method)) %>%
  mutate(rank = ifelse(method == "soft", NA, paste("k =", round(rank))))

resultdfinfo %>% 
  ggplot(aes(lambda, loss, color = method))+
  geom_point()+
  geom_line()+
  geom_label(data = resultdfinfo, mapping = aes(x = lambda - 5, y = ifelse(loss<0.3, loss-0.7, loss+0.7), label = rank), show.legend = FALSE, size = 2)+
  xlab(TeX("penatly factor (\\lambda)"))+
  ylab(TeX("$\\|\\sqrt{W} * (X - AB^T)\\|^2_F$"))+
  scale_x_continuous(breaks = seq(25, 200, 25))+
  theme(legend.position = "top")+
  labs(color = "")
  #geom_ribbon(aes(ymin = loss_min, ymax = loss_max), color = NA, alpha = 0.3)
ggsave("Plots/simulation_df.pdf", width = 6, height = 4)

