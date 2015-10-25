library(BoolNet)

args <- commandArgs(trailingOnly = TRUE)
nameIn <- args[1]
nameOut <- args[2]

net <- loadNetwork(nameIn)
attr <- getAttractors(net)

write(c(net$genes, "Basin"), file=nameOut, ncolumns=length(net)+1, sep="\t")
for (i in attr$attractor) {
  write(  c(
    intToBits(i$involvedStates)[1:length(net)], i$basinSize
    ), file=nameOut, ncolumns=length(net)+1, sep="\t", append = TRUE)
}


