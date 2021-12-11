from copy import deepcopy
from typing import List, Tuple
import heapq

import numpy as np
import torch

import numpy as np
from tqdm import tqdm

from .attacker import Attacker

from torch.nn.functional import softmax

class InputReduction(Attacker):
    """
    Runs the input reduction method from [Pathologies of Neural Models Make Interpretations
    Difficult](https://arxiv.org/abs/1804.07781), which removes as many words as possible from
    the input without changing the model's prediction.
    """
    def __init__(self,
                 model,
                 criterion,
                 tokenizer,               
                 beam_size = 3,
                 show_progress=True,
                 **kwargs):
        super().__init__(model, criterion, tokenizer, show_progress, **kwargs)
        # Hyperparameters
        self.beam_size = beam_size

    def attack(
        self, 
        test_dataloader,
        input_field_to_attack: str = "tokens",
        grad_input_field: str = "grad_input_1",
        ignore_tokens: List[str] = None,
        target = None,
    ):
        if target is not None:
            raise ValueError("Input reduction does not implement targeted attacks")
        ignore_tokens = ["@@NULL@@"] if ignore_tokens is None else ignore_tokens
        #ignore_tokens = self.special_tokens if ignore_tokens is None else ignore_tokens
        #print(self.special_tokens)
                          
        iterator = tqdm(test_dataloader) if self.show_progress else test_dataloader

        result = dict()
        for batch in iterator:

            self.batch_output = []
            grad_dict, _ = self._get_gradients(batch)                            
            #print(grad_dict[grad_input_field])
            self.batch_output.append(grad_dict[grad_input_field])
            batch_output = self.update_output()            
            #print(batch_output[0]['label'])
            #print(np.argmax(batch_output[0]['prob']))
            
            #print(labels)
            result["original_tokens"] = deepcopy(batch_output[0]["tokens"])
            result["final_tokens"] = []           
            result["final_tokens"].append(
                self._attack_instance(
                    test_dataloader, batch, input_field_to_attack, grad_input_field, ignore_tokens , batch_output[0]
                )
            )          
        #print(result)      
        return {"final": result["final_tokens"], "original": result["original_tokens"]}

    def _attack_instance(
        self,
        test_dataloader, # inputs
        instance, # batch in our case
        input_field_to_attack: str,
        grad_input_field: str,
        ignore_tokens: List[str],
        batch_output,
    ):        
        # Save fields that must be checked for equality      
        # text_field has field probably token by token
        # fields to compare is where to look
        fields_to_compare = {
            key: instance[key]
            for key in instance.keys()
            if key not in test_dataloader
            and key != input_field_to_attack
            and key != "input_ids"
            and key != "attention_mask"
        }
        
        initial_label = batch_output['label']

        # Set num_ignore_tokens, which tells input reduction when to stop
        # We keep at least one token for input reduction on classification/entailment/etc.
        if "tags" not in instance:
            # let's add an average one log length ...
            num_ignore_tokens = 3
            tag_mask = None       

        instance.update(batch_output)
        text_field= instance[input_field_to_attack]  # type: ignore
        #text_field = tokens
        current_tokens = deepcopy(text_field)
        print(f'Initial length: {len(current_tokens)}')
        #current_tokens = []
        #current_tokens.append(text_field) # IT CAN WORK IF YOU GET TEXT AS ARRAY?
        candidates = [(instance, -1)]
        # keep removing tokens until prediction is about to change
        #print(len(current_tokens))        
        #print(candidates)
        #print(len(current_tokens) > num_ignore_tokens and candidates)
        mystate = []
        mycount = 0 
        noChange = False       
        while len(current_tokens) > num_ignore_tokens and candidates:
       
            #if len(mystate) > 3:
            #    last = mystate.pop()[1]
            #    last2nd = mystate.pop()[1]
            #    last3rd = mystate.pop()[1]
            #    if last == last2nd and last == last3rd:
            #        noChange = True

            #if noChange:
            #    break
          
            # sort current candidates by smallest length (we want to remove as many tokens as possible)
            def get_length(input_instance):
                input_text_field = input_instance[input_field_to_attack]  # type: ignore
                return len(input_text_field)

            candidates = heapq.nsmallest(self.beam_size, candidates, key=lambda x: get_length(x[0]))            
            beam_candidates = deepcopy(candidates)
            candidates = []
            #print(f'This turn we use {len(beam_candidates)} candidates')            
            for beam_instance, smallest_idx in beam_candidates:        
                #print(f'This candidate current token s length is {len(beam_instance[input_field_to_attack])}')  
                #print(beam_instance)      
                # get gradients and predictions                
                grads, outputs = self._get_gradients(beam_instance)                
                #_, outputs = self.batch_output
                self.batch_output.append(grads[grad_input_field])
                # relabel beam_instance since last iteration removed an input token
                beam_output = self.update_output()
                #print(beam_output)
                beam_instance.update(beam_output[0])
                if initial_label != beam_instance["label"]:
                    #print('jumped..')
                    continue
                
                #print(f'Current beam label is {beam_instance["label"]}')
                #for output in outputs:
                #    if isinstance(outputs[output], torch.Tensor):
                #        outputs[output] = outputs[output].detach().cpu().numpy().squeeze().squeeze()
                #    elif isinstance(outputs[output], list):
                #        outputs[output] = outputs[output][0]


                # find a way to apply predictions to labeled instances
                # Check if any fields have changed, if so, next beam
                #if "tags" not in beam_instance:
                # relabel beam_instance since last iteration removed an input token
                #beam_instance = self._predictions_to_labeled_instances(
                #    beam_instance
                #)[0]
                #if any(beam_instance[field] != fields_to_compare[field] for field in fields_to_compare):
                #    continue
                
               

                # remove a token from the input
                text_field = beam_instance[input_field_to_attack]  # type: ignore
                #text_field = beam_output[0]['tokens']
                current_tokens = deepcopy(text_field)                   
                reduced_instances_and_smallest = _remove_one_token(
                    beam_instance,
                    input_field_to_attack,
                    grads[grad_input_field][0].detach().cpu().numpy(),
                    #grads,
                    ignore_tokens,
                    self.beam_size,
                    #beam_tag_mask,  # type: ignore
                )
                candidates.extend(reduced_instances_and_smallest)
            

            #mycount = mycount + 1
            #mystate.append((mycount,len(current_tokens)))

        return current_tokens  

    def _predictions_to_labeled_instances(self, instance):
        new_instance = deepcopy(instance)
        #label = np.argmax(instance["prob"])
        probs = softmax(self.batch_output[1], dim=-1)
        probs, labels = torch.max(probs, dim=-1)
        labels = labels.detach().cpu()
        #print(labels.item())
        new_instance['label'] = int(labels.item())
        return [new_instance]  

def _remove_token_with(copied_instance, remove_idx):

    keys_to_update = ['tokens','input_ids','attention_mask']
    for key in keys_to_update:
        if key in copied_instance:
            copied_field = copied_instance[key]
            if key == 'tokens':                
                # remove smallest
                inputs_before_smallest = copied_field[0:remove_idx]
                inputs_after_smallest = copied_field[remove_idx + 1 :]
                copied_field = inputs_before_smallest + inputs_after_smallest
                copied_instance[key] = copied_field
            else:
                # remove smallest
                copied_field = torch.cat((copied_field[:,:remove_idx], copied_field[:,remove_idx+1:]), axis = 1)                                
                #copied_field = torch.cat([copied_field[0][0:remove_idx], copied_field[0][remove_idx+1:]])
                copied_instance[key] = copied_field
        else:
            print(f'{key} value should exist in instance.')

    return copied_instance
    

def _remove_one_token(
    instance,
    input_field_to_attack: str,
    grads: np.ndarray,
    ignore_tokens: List[str],
    beam_size: int,
    #tag_mask: List[int],
):
    """
    Finds the token with the smallest gradient and removes it.
    """
    # Compute L2 norm of all grads.
    grads_mag = [np.sqrt(grad.dot(grad)) for grad in grads]

    # Skip all ignore_tokens by setting grad to infinity
    text_field = instance[input_field_to_attack]  # type: ignore
    for token_idx, token in enumerate(text_field):
        if token in ignore_tokens:
            grads_mag[token_idx] = float("inf")
    
    reduced_instances_and_smallest = []
    for _ in range(beam_size):
        # copy instance and edit later
        copied_instance = deepcopy(instance)        

        # find smallest
        smallest = np.argmin(grads_mag)        
        #print(text_field[smallest])
        if grads_mag[smallest] == float("inf"):  # if all are ignored tokens, return.
            break
        grads_mag[smallest] = float("inf")  # so the other beams don't use this token


        copied_instance = _remove_token_with(copied_instance,smallest)          
        #copied_instance.indexed = False
        #copied_text_field = copied_instance[input_field_to_attack]  # type: ignore
        #print(f'Removed length: {len(copied_text_field)}')       
        reduced_instances_and_smallest.append((copied_instance, smallest))

    return reduced_instances_and_smallest