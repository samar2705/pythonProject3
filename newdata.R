library(pcalg)
library(bnlearn)

# Load data into R
mydata <- read.csv("C:/Users/samar/OneDrive/Desktop/Final project/final_df.csv")


# convert data frame to matrix
mydata_matrix <- as.matrix(mydata)
mydata_df <- as.data.frame(mydata_matrix)


# Apply MMHC algorithm to learn the structure of the Bayesian network
bn.mmhc <- mmhc(mydata_df)
bn_hc <- hc(mydata_df, score = "bge")
# Extract the Markov blanket of the "target" variable
mb_nodes <- mb(bn.mmhc,"target")
# Learn a Bayesian network from the Markov blanket using MLE
bn_mb <- bn.fit(bn.mmhc, data=mydata_df, method="mle-g")


# Learn the structure of the network using GS
bn_gs <- mmhc(mydata_df)
# Fit the network using bn.fit and the BIC score
bn_mb_gs <- bn.fit(bn_gs, mydata_df)
# Extract the Markov blanket of node "target"
mb_b <- mb(bn_mb_gs, "target")




# define whitelist as matrix with two columns
whitelist <- matrix(ncol=2, byrow=TRUE, 
                    dimnames=list(NULL, c("from", "to")))
# set allowed edges
whitelist[1,] <- c("target", "target")

# Learn the structure of the network using pc
bn_pc <- mmhc(mydata_df)
# Fit the network using bn.fit and the BIC score
bn_mb_pc <- bn.fit(bn_pc, mydata_df)
# Extract the Markov blanket of node "target"
mb_p <- mb(bn_mb_pc, "target")



# Apply the MMHC algorithm to learn the structure of the Markov blanket network
nb_mb <- mmhc(mydata_df, whitelist = whitelist)
pc_mb <- pc.stable(mydata_df, whitelist = whitelist)
gs_mb <- gs(mydata_df, whitelist = whitelist)

print(bn_mb)
print(bn.mmhc)
print(nb_mb)
#print(mp_mb)
print(gs_mb)
print(pc_mb)

class(bn.mmhc)
class(bn_mb)
class(nb_mb)
#class(mp_mb)
class(gs_mb)
class(pc_mb)

#mb_ = mb(bn_mmhc , "target")
df_2 = mydata_df[,c(mb_p,"target")]

sum(is.na(df_2))

c1= pc.stable(df_2)
c2=mmhc(df_2)
c3=gs(df_2)
par(mfrow = c(1, 2))
graphviz.plot(c2)
graphviz.plot(bn_mb)
bn_mmhc_2=bn.fit(c2,df_2)

print(bn_mmhc_2)
class(bn_mmhc_2)

graphviz.plot(c2 , main = "mmhc")

## Subset the data frame using the Markov Blanket variables
#mb_features <- mydata_df[, mb_nodes]
## Print the values of the Markov Blanket features
#print(mb_features)

## Print the values of the nodes in c2
#for (node in c2$nodes) {
#  node_values <- mb_features[, mb_nodes]
#  print(node_values)
#}


#$mb: This represents the Markov blanket of the node. The Markov blanket of a node consists of the node's parents, its children, and any other variables that are direct causes or direct effects of the node
#$nbr: This represents the neighbors of the node. Neighbors are the nodes that are directly connected to the given node in the graph.
#$parents: This represents the parents of the node. Parents are the nodes that have a direct causal relationship with the given node.
#$children: This represents the children of the node. Children are the nodes that are directly affected by the given node.
print(node)
