import torch
from torch.nn.functional import softmax

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

class Attacker:
    """
    An `Attacker` will modify an input (e.g., add or delete tokens) to try to change an AllenNLP
    Predictor's output in a desired manner (e.g., make it incorrect).
    """

    def __init__(self,
                model,
                criterion,
                tokenizer,
                show_progress=True,
                **kwargs):

            """
            :param model: nn.Module object - can be HuggingFace's model or custom one.
            :param criterion: torch criterion used to train your model.
            :param tokenizer: HuggingFace's tokenizer.
            :param show_progress: bool flag to show tqdm progress bar.
            :param kwargs:
                encoder: string indicates the HuggingFace's encoder, that has 'embeddings' attribute. Used
                    if your model doesn't have '    input_embeddings' method to get access to encoder embeddings
            """

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            #self.device = torch.device('cpu')
            self.model = model.to(self.device)
            #self.model.eval()
            self.criterion = criterion
            self.tokenizer = tokenizer
            self.show_progress = show_progress
            self.kwargs = kwargs
            # to save outputs 
            self.batch_output = None

    def _get_gradients(self, batch):
        # set requires_grad to true for all parameters, but save original values to
        # restore them later
        original_param_name_to_requires_grad_dict = {}
        for param_name, param in self.model.named_parameters():
            original_param_name_to_requires_grad_dict[param_name] = param.requires_grad
            param.requires_grad = True
        embedding_gradients = []
        hooks = self._register_embedding_gradient_hooks(embedding_gradients)

        loss = self.forward_step(batch)

        self.model.zero_grad()
        loss.backward()

        for hook in hooks:
            hook.remove()

        grad_dict = dict()
        for idx, grad in enumerate(embedding_gradients):
            key = "grad_input_" + str(idx + 1)
            grad_dict[key] = grad
        
        # restore the original requires_grad values of the parameters
        for param_name, param in self.model.named_parameters():
            param.requires_grad = original_param_name_to_requires_grad_dict[param_name]
                    
        # this should give an output of grads and output!!1 
        # maybe we should give all not the first index!
        #return embedding_gradients[0]
        return grad_dict, self.batch_output[1]

    def _register_embedding_gradient_hooks(self, embedding_gradients):
        """
        Registers a backward hook on the
        Used to save the gradients of the embeddings for use in get_gradients()
        When there are multiple inputs (e.g., a passage and question), the hook
        will be called multiple times. We append all the embeddings gradients
        to a list.
        """

        def hook_layers(module, grad_in, grad_out):
            embedding_gradients.append(grad_out[0])

        backward_hooks = []
        embedding_layer = self.get_embeddings_layer()
        backward_hooks.append(embedding_layer.register_backward_hook(hook_layers))
        return backward_hooks

    def get_embeddings_layer(self):
        if hasattr(self.model, "get_input_embeddings"):
            embedding_layer = self.model.get_input_embeddings()
        else:
            encoder_attribute = self.kwargs.get("encoder")
            assert encoder_attribute, "Your model doesn't have 'get_input_embeddings' method, thus you " \
                "have provide 'encoder' key argument while initializing SaliencyInterpreter object"
            embedding_layer = getattr(self.model, encoder_attribute).embeddings
        return embedding_layer

    def colorize(self, results, initials):

        #special_tokens = self.special_tokens       
        template = '<span class="barcode"; style="color: black; background-color: #f5424e">{}</span>'
        template_not_selected = '<span class="barcode"; style="color: black; background-color:#bab3b3; text-decoration:line-through;">{}</span>'
        
        colored_string = ''      
        for idx, word in enumerate(initials):        
            if word in results:
                word = word.replace("Ġ", " ") if 'Ġ' in word else '' + word
                colored_string += template.format(word)    
            else:
                word = word.replace("Ġ", " ") if 'Ġ' in word else ' ' + word
                colored_string += template_not_selected.format(word)
        #colored_string += template.format(0, "    Label: {} |".format(instance['label']))
        #prob = instance['prob']
        #color = matplotlib.colors.rgb2hex(prob_cmap(prob)[:3])
        #colored_string += template.format(color, "{:.2f}%".format(instance['prob']*100)) + '|'
        return colored_string.replace("Ġ", " ")

    @property
    def special_tokens(self):
        """
        Some tokenizers don't have 'eos_token' and 'bos_token' attributes.
        So needed we some trick to get them.
        """
        if self.tokenizer.bos_token is None or self.tokenizer.eos_token is None:
            special_tokens = self.tokenizer.build_inputs_with_special_tokens([])
            special_tokens_ids = self.tokenizer.convert_ids_to_tokens(special_tokens)
            self.tokenizer.bos_token, self.tokenizer.eos_token = special_tokens_ids

        special_tokens = self.tokenizer.eos_token, self.tokenizer.bos_token
        return special_tokens

    def forward_step(self, batch):
        """  
        :param batch: batch returned by dataloader
        :return: torch.Tensor: batch loss
        """
        
        input_ids = batch.get('input_ids').to(self.device)
        attention_mask = batch.get("attention_mask").to(self.device)
        # TODO this part is manually updated, change this
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, adapter_names=['mlm-firewall','fanomaly','mlm-hdfs','hanomaly'])       
        #outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, adapter_names=['mlm','fanomaly'])       
        label = torch.argmax(outputs[0], dim=1)
        batch_losses = self.criterion(outputs[0], label)
        loss = torch.mean(batch_losses)

        self.batch_output = [input_ids, outputs[0]]
        #print(self.batch_output)

        return loss

    def update_output(self):
        """       
        :return: batch_output
        """
        #print(self.batch_output)
        input_ids, outputs, grads = self.batch_output

        probs = softmax(outputs, dim=-1)
        probs, labels = torch.max(probs, dim=-1)

        tokens = [
            self.tokenizer.convert_ids_to_tokens(input_ids_)
            for input_ids_ in input_ids
        ]

        embedding_grads = grads.sum(dim=2)
        # norm for each sequence
        norms = torch.norm(embedding_grads, dim=1, p=1)
        # normalizing
        for i, norm in enumerate(norms):
            embedding_grads[i] = torch.abs(embedding_grads[i]) / norm

        batch_output = []

        iterator = zip(tokens, probs, embedding_grads, labels)

        for example_tokens, example_prob, example_grad, example_label in iterator:
            example_dict = dict()
            # as we do it by batches we has a padding so we need to remove it
            example_tokens = [t for t in example_tokens if t != self.tokenizer.pad_token]
            example_dict['tokens'] = example_tokens
            example_dict['grad'] = example_grad.cpu().tolist()[:len(example_tokens)]
            example_dict['label'] = example_label.item()
            example_dict['prob'] = example_prob.item()
            batch_output.append(example_dict)
        
        return batch_output    