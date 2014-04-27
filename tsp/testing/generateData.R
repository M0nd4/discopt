## Generate some points for test graph
xs <- rpois(10,5)
ys <- rpois(10,5)
dat <- cbind(xs, ys)

write.table(dat, file="~/class/discopt/tsp/testing/testGraph.txt", row.names=FALSE,
            col.names = FALSE)
