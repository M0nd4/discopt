library(ggplot2)
library(grid)

## function to read data and solutions and order data by solution
reload <- function(dataset, soln) {
    testdir <- "~/class/discopt/tsp/testing/"
    datadir <- "~/class/discopt/tsp/data/"
    # read in data
    input <- paste0(datadir, filename)
    dat <- read.table(input, header=FALSE, fill=T)
    N <- dat[1,1]
    cities <- dat[2:nrow(dat),]
    # read in solution path
    ss <- paste0("/solutions/", soln)
    path <- scan(paste0(testdir, ss))
    p_len <- path[1]
    path <- path[2:length(path)]
    # order data by path
    cities$num <- c(0:(length(path)-1))
    cities <<- cities[order(match(cities$num, path)),]
}

####################################################
## Part 1, Goal: 430
filename = "tsp_51_1"

## Greedy solution
soln = "greedy_51.data"
reload(filename, soln)

## 3opt
soln = "3opt_51.data"
reload(filename, soln)

## threeOpt, different starts
soln = "threeOpt_51.data"
reload(filename, soln)

## kopt
soln = "kopt_40_51.data"
reload(filename, soln)

## anneal
soln = "anneal_51.data"
reload(filename, soln)

## plot
dev.new()
ggplot(cities, aes(V1, V2, label = num)) + geom_point(col = "blue") +
    geom_path(lty = 2) + geom_text()

####################################################
## Part 2
filename = "tsp_100_3"

## 2opt
soln = "2opt_100.data"
reload(filename, soln)

## 3opt
soln = "3opt_100.data"
reload(filename, soln)

## threeOpt
soln = "threeOpt_100.data"
reload(filename, soln)

## kopt
soln = "kopt_100.data"
reload(filename, soln)

## anneal
soln = "anneal_100.data"
reload(filename, soln)

## plot
dev.new()
ggplot(cities, aes(V1, V2, label = num)) + geom_point(col = "blue") +
    geom_path(lty = 2) + geom_text()

####################################################
## Part 3
filename = "tsp_200_2"

## greedy
soln = "greedy_200.data"
reload(filename, soln)

## threeOpt
soln = "threeOpt_200.data"
reload(filename, soln)

## kopt
soln = "kopt_99_200.data"
reload(filename, soln)

## anneal
soln = "anneal_200.data"
reload(filename, soln)

## Plot
dev.new()
ggplot(cities, aes(V1, V2, label = num)) + geom_point(col = "blue") +
    geom_path(lty = 2) # + geom_text()

####################################################
## Part 4
filename = "tsp_574_1"

## Greedy solution
soln = "greedy_574.data"
reload(filename, soln)

## 3opt
soln = "3opt_574.data"
reload(filename, soln)

## threeOpt
soln = "threeOpt_574.data"
reload(filename, soln)

## threeOpt random start
soln = "threeOpt_random_574.data"
reload(filename, soln)

## kopt
soln = "kopt_574.data"
reload(filename, soln)

## anneal
soln = "neighbor_anneal_574.data"
reload(filename, soln)
## plot
ggplot(cities, aes(V1, V2, label = num)) + geom_point(col = "blue") +
    geom_path(lty = 2) # + geom_text()

####################################################
## Part 5
filename = "tsp_1889_1"

## greedy
soln = "greedy_1889.data"
reload(filename, soln)

## 3opt
soln = "3opt_1889.data"
reload(filename, soln)

## kopt
soln = "kopt_1889.data"
reload(filename, soln)

## rev_trans_anneal
soln = "rev_trans_anneal_1889.data"
reload(filename, soln)

## plot
ggplot(cities, aes(V1, V2, label = num)) + geom_point(col = "blue") +
    geom_path(lty = 2) # + geom_text()

####################################################
## Part 6
filename = "tsp_33810_1"
xs <- c(0.0, 17447.5, 34895.0, 52342.50000000001, 69790.0, 87237.5, 104685.00000000001, 122132.50000000001, 139580.0, 157027.5, 174475.0, 191922.50000000003, 209370.00000000003, 226817.5, 244265.00000000003, 261712.5, 279160.0, 296607.50000000006, 314055.0, 331502.5, 348950.0, 366397.5, 383845.00000000006, 401292.50000000006, 418740.00000000006, 436187.5, 453635.0, 471082.50000000006, 488530.00000000006, 505977.50000000006, 523425.0, 540872.5, 558320.0, 575767.5, 593215.0000000001, 610662.5, 628110.0, 645557.5, 663005.0, 680452.5000000001, 697900.0)

ys <- c(0.0, 15122.5, 30245.0, 45367.50000000001, 60490.0, 75612.5, 90735.00000000001, 105857.50000000001, 120980.0, 136102.5, 151225.0, 166347.5, 181470.00000000003, 196592.5, 211715.00000000003, 226837.5, 241960.0, 257082.50000000003, 272205.0, 287327.5, 302450.0, 317572.5, 332695.0, 347817.50000000006, 362940.00000000006, 378062.5, 393185.0, 408307.5, 423430.00000000006, 438552.50000000006, 453675.0, 468797.5, 483920.0, 499042.50000000006, 514165.00000000006, 529287.5, 544410.0, 559532.5, 574655.0, 589777.5, 604900.0)

# init
soln = "neighbor_33810.data"
reload(filename, soln)

# anneal
soln = "anneal_33810.data"
reload(filename, soln)

# anneal with neighbors
soln = "neighbor_anneal_33810.data"
reload(filename, soln)

## plot
dev.new()
ggplot(cities, aes(V1, V2, label = num)) + geom_point(col = "blue") +
    geom_path(lty = 2) #+ geom_vline(xintercept = xs, col = "red") +
    geom_hline(yintercept = ys, col = "red")

## ggsave(filename = "~/class/discopt/tsp/testing/solutions/155.png")
