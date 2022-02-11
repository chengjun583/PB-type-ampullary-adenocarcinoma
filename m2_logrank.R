# Do log rank test and plot KM curves

rm(list = ls())
ptm = proc.time()

library(survival)

myData = read.table("res_m1.txt");
surTime = myData[[1]];
death = myData[[2]];
mySurv = Surv(surTime, death);
group = myData[[3]]

# Log-rank test
log1 = survdiff(mySurv ~ group)
p = pchisq(log1$chisq, 1, lower.tail=FALSE)
print(paste0('Log-rank test P value: ', p))

# Survival curve
ng = length(unique(group))
n1 = sum(group==1)
leName1 = paste("Treatments for PAC(", n1, ")", sep = "")
n2 = sum(group==2)
leName2 = paste("Treatments for CHOL(", n2, ")", sep = "")

fit = survfit(mySurv ~ group)
fname = 'res_m2_KMCurve.png'

png(filename = fname, width = 2.8, height = 2.8,
	units = "in", res = 300, pointsize = 7)
plot(fit, mark.time=TRUE, xlab = "Months", ylab = "Survival", lty = 1:ng,
	col = 1:ng, cex = 0.5)
grid()
#legend(x = "topright", legend = c(leName1, leName2), lty = 1:ng, col = 1:ng, cex = 0.65)
# text(10, 0.1, paste("p=", formatC(p, format="g", digits = 3), sep = ""), pos = 4, cex = 1)
dev.off()

print(summary(fit, c(0, 10, 20, 30, 40)))

print(proc.time() - ptm)

