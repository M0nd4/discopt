args<-commandArgs(TRUE)
filename = args[1]
## sprintf("Adding one to nodes in %s", filename)
## sprintf("Output file is 'temp' in this directory")
graph <- read.table(filename, header = FALSE)

graph[2:nrow(graph),] = graph[2:nrow(graph), ] + 1

write.table(graph, "~/class/discopt/coloring/testing/temp", row.names = FALSE, col.names = FALSE)
