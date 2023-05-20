folder <- "~/Enrique/Repositories/ProbExplainer/data"
N_EXPERTS = 42
N_CELLS = 320


annotations <- gardenr::get_all_labels(folder)
# SAVE DATASETS
for (i in seq(N_EXPERTS)) {
  local_dataset = annotations %>% filter(annotator == i)
  file_dataset = paste("~/Enrique/Repositories/ProbExplainer/expert_datasets/dataset_",i,".csv",sep="")
  write.csv(local_dataset, file_dataset)
}

# PREPROCESS DATA
annotations = droplevels(annotations[0:(N_EXPERTS*N_CELLS),])
for (i in dimnames(annotations)[[2]]) {
  annotations[,i] = factor(str_replace_all(annotations[,i] ,' ','_'))
}
annotations = annotations[!grepl("uncharacterized", annotations$F6),]
annotations = droplevels(annotations)

for (i in seq(N_EXPERTS)) {
  local_dataset = annotations %>% filter(annotator == i)
  local_dataset = local_dataset[,c("F1","F2","F3","F4","F5")]
  dag = bnclassify::tan_hc("F5",local_dataset,k=5)
  dag_string = bnclassify::modelstring(dag)
  dag_string = str_replace_all(dag_string," ", "") 
  dag_bnlearn = bnlearn::model2network(dag_string,ordering = c("F1","F2","F3","F4","F5"))

  fitted = bnlearn::bn.fit(dag_bnlearn, local_dataset, method = "bayes")
  file = paste("~/Enrique/Repositories/ProbExplainer/expert_networks/network_",i,".bif",sep="")
  bnlearn::write.bif(file, fitted)
}
