# multivariate Cox regression

mydata = read.table("res_m3_data.txt", header = FALSE)
mydata = data.matrix(mydata)
s1 = nrow(mydata)
s2 = ncol(mydata)
surTime = mydata[, 1]
death = mydata[, 2]
x = mydata[, 3:s2]


my.surv <- Surv(surTime, death)

s2_x = ncol(x)
rname = c("treatment", "sex", 'diff')
cname = c("exp(coef)", "exp(-coef)", "lower .95", "upper .95", "p")
resMulti = matrix(nrow = s2_x, ncol = 5, dimnames = list(rname, cname))

coxph.fit <- coxph(my.surv ~ x, method="breslow")
mysum = summary(coxph.fit)
resMulti[, 1:4] = mysum$conf.int
resMulti[, 5] = mysum$coefficients[, 5]

print(resMulti)
save(resMulti, file = "res_m4_multiCox.RData")