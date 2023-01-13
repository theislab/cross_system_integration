# %%
import scanpy as sc
import pandas as pd
import os
import obonet
import networkx

# %%
def get_parents(graph, cell_type_name, max_n_parents_per_gen=10, verbose=False):
    """Calculates all parent cell types of a given cell type, going up (coarser)
    in the graph generation by generation. Parent search stops either when
    there are no more parents (not sure if that ever happens, as the graph
    is not acyclic), or until more than max_n_parents_per_gen are found
    at a specific generation.
    Arguments;
    graph: obonet graph that contains cell type ontology
    cell_type_name: string of cell type name for which we want to find the
        parents based on the ontology graph.
    max_n_parents_per_gen: integer, specifies at which point to stop searching 
        for parents at even lower levels in the ontology. As soon as more than
        max_n_parents_per_gen parents are found at a certain level, the search
        is stopped and no more levels are added.
    verbose: Boolean, whether to print info or not.
    
    Returns:
    Dictionary with node names of parents for every generation up the ontology
    graph.
    
    Example:
    ct_parents = dict()
    for ct_name in ct_names:
        ct_parent_names = get_parents(graph=graph, cell_type_name=ct_name, max_n_parents_per_gen=20, verbose=False)
        ct_parents[ct_name] = ct_parent_names
    # store in dataframe:
    ct_parent_df = pd.DataFrame(ct_parents)
    """
    id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
    name_to_id = {data['name']: id_ for id_, data in graph.nodes(data=True) if 'name' in data}
    if verbose:
        print(f"Working on cell type {cell_type_name}...")
    has_parents = True
    node_names_per_gen = dict()
    gen_has_parents = dict()
    generation=-1
    node_names_per_gen[0] = [cell_type_name]

    while has_parents:
        generation += 1
        node_names_per_gen[generation + 1] = list()
        gen_has_parents[generation] = False
        # loop through all nodes at this level (in case cell type has multiple parents)
        for node_name in node_names_per_gen[generation]:
            successors = [succ for succ in list(graph.successors(name_to_id[node_name])) if succ.startswith("CL")]
            if len(successors) > 0:
                node_names_per_gen[generation + 1] += [id_to_name[suc] for suc in successors]
                gen_has_parents[generation] = True
        has_parents = gen_has_parents[generation]
        if verbose:
            print(generation, has_parents, [node_name for node_name in node_names_per_gen[generation + 1]])
        if len(node_names_per_gen[generation + 1]) > max_n_parents_per_gen:
               if verbose:
                    print(f"More than {max_n_parents_per_gen} parents found. Stopping at generation {generation}.")
               break
    return node_names_per_gen


# %%
def flatten_nested_list(nested_list):
    """Turns nested list into single flat list. Returns flat list."""
    return [item for sublist in nested_list for item in sublist]


# %%

# %%
def get_finest_common_labels(graph, ct_names, ct_parent_df, max_rounds=20, verbose=True):
    """Finds the finest possible mapping of original labels, such that none
    of the final labels is a parent of any of the other final labels.
    
    Arguments:
    graph: obonet graph of cell type ontology to use
    ct_names: list of cell type names to start with
    ct_parent_df: pandas dataframe containing information about parents for 
        every cell type in ct_names, as generated with get_parents function
        according to example given in function documentation.
    max_rounds: integer. maximum number of rounds to go through re-mapping loop. Loop 
        will stop automatically before that round if no more parent-child relations
        are found. This argument is mostly to prevent infinite loops if something 
        is wrong.
    verobse: Boolean, whether to print info or not.
    
    Returns:
    Dictionary with mapping of original names (i.e. ct_names) to proposed
    final labels.
    """
    # generate id to name and name to id mapping based on ontology:
    id_to_name = {id_: data.get('name') for id_, data in graph.nodes(data=True)}
    name_to_id = {data['name']: id_ for id_, data in graph.nodes(data=True) if 'name' in data}
    # create first draft dictionary of cell type to final mapping:
    ct_to_integration_name = {ct:ct for ct in ct_names}
    integration_labels_try = flatten_nested_list(ct_parent_df.loc[0,:].tolist())
    parent_child_in_labels = True
    round=0
    # As long as we still find parent-child pair in our final labels, the 
    # parent_child_in_label will be kept True and we'll keep going through
    # this loop
    while parent_child_in_labels and round<max_rounds:
        if verbose:
            print(round)
        # check that we have parent information for every cell id:
        for ct_name in integration_labels_try:
            if ct_name not in ct_parent_df.columns:
                if verbose:
                    print(f"NOTE: Adding {ct_name} and its parents as a column to ct_parent_df.")
                ct_parents = get_parents(graph, ct_name)
                ct_parent_df.loc[ct_parents.keys(), ct_name] = list(ct_parents.values)
        # start looping through integration label pairs, and set 
        # parent_child_in_labels to True as soon as we find a pair of which
        # one is the parent of the other
        parent_child_in_labels = False
        # for every proposed integration label, check if it has parents
        # in the remaining proposed integration labels
        # track for every ct_name if it has already been changed this round;
        # if so, don't loop through it again (as that will make us toggle 
        # between states indefinitely.
        # initiate variable that tracks if this label needs to change
        int_names_changed_this_round = dict()
        for ct_name_1 in integration_labels_try:
            ct_name_1_changed = False
            # get parents of ct_name_1
            ct_name_1_parents = flatten_nested_list(ct_parent_df.loc[1:,ct_name_1].dropna().tolist())
            # and every proposed integration label id again (i.e. for every pair)
            for ct_name_2 in integration_labels_try:
                # if a parent of ct_name_1 corresponds to ct_name_2, 
                # we need to remove ct_name_1 (the finer, more granular id)
                # from our proposed integration labels:
                if ct_name_2 in ct_name_1_parents:
                    parent_child_in_labels = True
                    # check if ct_name_2 was already changed this round
                    # if so, incorporate that change into the updates below
                    if ct_name_2 in int_names_changed_this_round.keys():
                        new_name = int_names_changed_this_round[ct_name_2]
                    else:
                        new_name = ct_name_2
                    # if ct_name_2 was in the original cell type labels (and 
                    # is not yet a coarsified version of those)
                    # map it to the coarser ct_name_2
                    if ct_name_1 in ct_to_integration_name.keys():
                        if verbose:
                            print(f"{ct_name_1} will be set to {new_name}.")
                        ct_to_integration_name[ct_name_1] = new_name
                        int_names_changed_this_round[ct_name_1] = new_name
                        ct_name_1_changed = True 
                        # toggle between states back and forth
                    # if ct_name_1 is already a coarsified version of one 
                    # of the original ct labels, map that original ct label
                    # to the coarser ct_name_2 instead of the finer ct_name_1
                    if ct_name_1 in ct_to_integration_name.values():
                        # loop through the mapping dictionary to identify
                        # which exact key-value pair has ct_name_1 as value
                        for original_ct_name, ct_name_for_integration in ct_to_integration_name.items():
                            # if the integration label corresponds to our ct_name_1
                            if ct_name_for_integration == ct_name_1: # and ct_id != ct_id_1:
                                if verbose:
                                    print(f"Coarsifying '{original_ct_name}'s mapping from '{ct_name_1}' to '{new_name}'")
                                # coarsify the mapping
                                ct_to_integration_name[original_ct_name] = new_name
                                ct_name_1_changed = True
                if ct_name_1_changed:
                    break
            if not ct_name_1_changed:
                if verbose:
                    print(f"{ct_name_1} will stay as is.")
        integration_labels_try = sorted(set(ct_to_integration_name.values()))
        round += 1
        if verbose:
            print(f"\n\n NEW INTEGRATION LABEL SET: {integration_labels_try}\n\n")
    return ct_to_integration_name
