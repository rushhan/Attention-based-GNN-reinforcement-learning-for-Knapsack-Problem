import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateCVRP(NamedTuple):
    # Fixed input
    weight: torch.Tensor  # Depot + loc
    value: torch.Tensor
    null_weight:torch.Tensor
    null_value : torch.Tensor
    used_capacity: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    total_weight: torch.Tensor
    total_value: torch.Tensor
    i: torch.Tensor  # Keeps track of step
    VEHICLE_CAPACITY:torch.Tensor

    @property
    def visited(self):
            return self.visited_


    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                used_capacity=self.used_capacity[key],
                visited_=self.visited_[key],  # this will give us the last node visited
                total_weight=self.total_weight[key],
                total_value=self.total_value[key],
                
            )
        return super(StateCVRP, self).__getitem__(key)

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, visited_dtype=torch.uint8):

        weight = input['weight']
        value = input['value']
        null_value= input['null_value']
        null_weight= input['null_weight']
        VEHICLE_CAPACITY=input['VEHICLE_CAPACITY']
        #print ("the ve",VEHICLE_CAPACITY[0][0])
        batch_size, n_loc, = weight.size()
        return StateCVRP(
            weight=torch.cat((null_weight,weight),dim=-1),
            value=torch.cat((null_value,value),dim=-1),
            used_capacity=weight.new_zeros(batch_size, 1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=weight.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, dtype=torch.long, device=weight.device),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, n_loc+1,
                    dtype=torch.uint8, device=weight.device
                
               
                )  # Ceil
            ),
            total_weight=torch.zeros(batch_size, 1, device=weight.device),
            total_value=torch.zeros(batch_size, 1, device=weight.device),
            i=torch.zeros(1, dtype=torch.int64, device=weight.device),  # Vector with length num_steps
            null_weight=null_weight,
            null_value=null_value,
            VEHICLE_CAPACITY=VEHICLE_CAPACITY,
        )

    def get_final_cost(self):


        return (self.total_weight -self.total_value).squeeze(dim=-1),None

    def update(self, selected):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        selected = selected[:, None]  # Add dimension for step
        prev_a = selected
        n_loc = self.weight.size(-1)  # Excludes depot
        #print ("got here 1")
        #print (selected,self.ids)
        # Add the length
        cur_weight = self.weight[self.ids, selected]
        cur_val = self.value[self.ids,selected]
        # cur_coord = self.coords.gather(
        #     1,
        #     selected[:, None].expand(selected.size(0), 1, self.coords.size(-1))
        # )[:, 0, :]
        total_weight = self.total_weight + cur_weight # (batch_dim, 1)
        total_value = self.total_value+cur_val
        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        #selected_demand = self.demand.gather(-1, torch.clamp(prev_a - 1, 0, n_loc - 1))
        selected_weight = cur_weight# self.weight[self.ids, torch.clamp(prev_a - 1, 0, n_loc)]
        #print("the selected weight is this one",selected_weight)

        # Increase capacity if depot is not visited, otherwise set to 0
        #used_capacity = torch.where(selected == 0, 0, self.used_capacity + selected_demand)
        used_capacity = (self.used_capacity + selected_weight) .float()
        #print ("the used capacity is",used_capacity,selected_weight)
        
        visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        

        return self._replace(
            prev_a=prev_a, visited_=visited_,
            total_weight=total_weight,total_value=total_value,  i=self.i + 1,
            used_capacity =used_capacity
        )

    
    
    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        
        visited_loc = self.visited_[:, :, 1:]

        
        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = (self.weight[self.ids, :][:,:,1:] + self.used_capacity[:, :, None] > self.VEHICLE_CAPACITY[0,0]) # JEED TO Chech the dims
        # Nodes that cannot be visited are already visited or too much demand to be served now
        #print(self.ids)

        #print (exceeds_cap.shape)
        #sys.exit()

        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot =  ((mask_loc == 0).int().sum(-1) > 0)
        return torch.cat((mask_depot[:, :, None], mask_loc), -1)

    def construct_solutions(self, actions):
        return actions