library(dplyr)
library(ggplot2)
library(latex2exp)
library(scales)

#compare hard impute
resultsim = readRDS("Fits/simulation_hard_adaptive.rds")
ks = unique(resultsim$k)
resultsim %>% group_by(weights, prop, k) %>% filter(k != 5) %>%
  mutate(k = factor(k, levels = ks, labels = paste("k =", ks))) %>%
  summarise(loss_sd = sd(loss)/sqrt(30), loss = mean(loss))  %>%
ggplot(aes(prop, loss, color = weights))+
  geom_line()+
  geom_point()+
  xlab("proportion of missing values")+
  ylab("test error")+
  geom_ribbon(aes(ymin = loss - loss_sd, ymax = loss + loss_sd, fill = weights), alpha = 0.3, color = NA)+
  facet_wrap(~k, scales = "free", nrow = 1)




  
