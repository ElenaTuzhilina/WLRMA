library(dplyr)
library(ggplot2)
library(latex2exp)
library(scales)
library(viridis)

resultw = readRDS("Fits/real_data_hard_weights_info.rds") %>%
  mutate(weights = factor(ifelse(weights == "unit", "binary", weights), levels = c("binary", "pre-computed", "adaptive")))
nfold = max(resultw$fold)
resultw %>% group_by(k, weights) %>%  #filter(fold == 2) %>%
  summarise(loss_se = sd(loss)/sqrt(nfold), loss = mean(loss)) %>%
  ggplot(aes(k, loss, color = weights))+
  geom_line()+
  geom_point()+
  geom_ribbon(aes(ymin = loss - loss_se, ymax = loss + loss_se, fill = weights), color = NA, alpha = 0.2)+
  xlab("solution rank (k)")+
  ylab("cross-validation score")
ggsave("Plots/real_data_weights.pdf", width = 4, height = 3)

resultws2 = readRDS("Fits/real_data_hard_weights_s2.rds") 
gg_color_hue = function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
resultws2 %>%
  ggplot(aes(s2, fill = weights))+
  geom_histogram(position="identity", alpha = 0.6)+
  scale_fill_manual(values = gg_color_hue(3)[c(3,2)])+
  theme(legend.position = "none")+
  xlab(TeX("estimated user variance ($\\sigma^2$)"))
ggsave("Plots/real_data_weights_s2.pdf", width = 3, height = 3)

resultl = readRDS("Fits/real_data_prob_info.rds")
nfold = max(resultw$fold)
resultl %>% group_by(k) %>%  
  summarise(loss_se = sd(loss)/sqrt(nfold), loss = mean(loss),
            auc_se = sd(auc)/sqrt(nfold), auc = mean(auc)) %>%
  ggplot(aes(k, auc))+
  geom_line()+
  geom_point()+
  geom_ribbon(aes(ymin = auc - auc_se, ymax = auc + auc_se), color = NA, alpha = 0.2)+
  xlab("solution rank (k)")+
  ylab("cross-validation auc")
ggsave("Plots/real_data_logit.pdf", width = 5, height = 3)

resultl %>% group_by(k) %>%  
  summarise(loss_se = sd(loss)/sqrt(nfold), loss = mean(loss),
            auc_se = sd(auc)/sqrt(nfold), auc = mean(auc)) %>%
  ggplot(aes(k, loss))+
  geom_line()+
  geom_point()+
  geom_ribbon(aes(ymin = loss - loss_se, ymax = loss + loss_se), color = NA, alpha = 0.2)+
  xlab("solution rank (k)")+
  ylab("cross-validation auc")

resultlp = readRDS("Fits/real_data_hard_logit_probs.rds") %>%
  mutate(observed = factor(observed))
ggplot(resultlp, aes(movie, user, fill= probability)) + 
  geom_tile() +
  scale_fill_viridis(discrete = FALSE, limits=c(0, 1))+
  theme_minimal()
ggsave("Plots/real_data_logit_heatmap1.pdf", width = 4, height = 3)

ggplot(resultlp, aes(movie, user, fill = observed)) + 
  geom_tile() +
  scale_fill_viridis(discrete = TRUE, breaks = c(1,0))+
  theme_minimal()
ggsave("Plots/real_data_logit_heatmap2.pdf", width = 4, height = 3)


roc(resultlp$observed, resultlp$probability) %>%
  ggroc()
ggsave("Plots/real_data_logit_roc.pdf", width = 3, height = 3)
