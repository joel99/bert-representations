#%%
# Multitask models are typically motivated as trunk/branch, because one monolithic architecture will make it difficult to learn representations relevant to every downstream task.
# Given sufficient data/compute, we could learn individual branches.
# But we don't generally have this. Given limited model capacity for multitask learning, what is the best way to combine the tasks?
# Specifically, which combination of leads to best overall downstream task performance, given a fixed number of "heads" (transformer branches)

