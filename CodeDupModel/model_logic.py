from torch.autograd import Function
import torch.nn as nn
import torch
from config import Config


class ReverseLayerF(Function):
    """
    multiplies the gradient by a negative value during backpropagation -> adversarial training
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class LanguageTransformation(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(LanguageTransformation, self).__init__()
        self.transformation = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        return self.transformation(x)


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, languages):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.languages = languages

        print(f"Languages: {languages}")
        print(f"Number of languages: {len(languages)}")

        self.domain_classifier = nn.Sequential(nn.Dropout(p=config.dropout_rate), nn.Linear(config.hidden_size, len(languages)))
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.cycle_criterion = nn.L1Loss()

        self.transformations = nn.ModuleDict()
        for lang1 in self.languages:
            for lang2 in self.languages:
                if lang1 != lang2:
                    key = f"{lang1}_to_{lang2}"
                    self.transformations[key] = LanguageTransformation(config.hidden_size, config.intermediate_size, config.dropout_rate)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, positive_input_ids, negative_input_ids, labels, negative_labels, domain_labels, alpha):
        batch_size, _ = input_ids.size()

        # Concatenating input IDs for anchor, positive, and negative examples
        concatenated_input_ids = torch.cat((input_ids, positive_input_ids, negative_input_ids), 0)

        # Obtain the output embeddings from the encoder
        encoder_outputs = self.encoder(concatenated_input_ids, attention_mask=concatenated_input_ids.ne(1))[1]

        # Split the outputs back into separate batches
        anchor_outputs, positive_outputs, negative_outputs = encoder_outputs.split(batch_size, 0)

        # Calculate similarity scores for contrastive loss
        positive_similarity = (anchor_outputs * positive_outputs).sum(-1)
        negative_similarity = (anchor_outputs * negative_outputs).sum(-1)

        # Calculate contrastive loss
        probabilities = torch.softmax(torch.cat((positive_similarity[:, None], negative_similarity[:, None]), -1), -1)
        loss = -torch.log(probabilities[:, 0] + 1e-10).mean()

        # Apply transformations and compute domain logits only for anchor outputs
        all_transformed_vectors = torch.zeros_like(anchor_outputs)
        all_cycle_vectors = torch.zeros_like(anchor_outputs)

        for lang_id, lang in enumerate(self.languages):
            lang_indices = (domain_labels == lang_id).nonzero(as_tuple=True)[0]
            for target_lang_id, target_lang in enumerate(self.languages):
                if lang != target_lang:
                    # Apply transformation
                    transformed_key = f"{lang}_to_{target_lang}"
                    transformed_vectors = self.transformations[transformed_key](anchor_outputs[lang_indices])

                    # Apply cycle transformation
                    cycle_transformed_key = f"{target_lang}_to_{lang}"
                    cycle_transformed_vectors = self.transformations[cycle_transformed_key](transformed_vectors)

                    # Assign vectors to corresponding positions
                    all_transformed_vectors[lang_indices] = transformed_vectors
                    all_cycle_vectors[lang_indices] = cycle_transformed_vectors

        # Normalize vectors
        normalized_transformed_vectors = torch.nn.functional.normalize(all_transformed_vectors, p=2, dim=-1)
        normalized_cycle_vectors = torch.nn.functional.normalize(all_cycle_vectors, p=2, dim=-1)

        # Calculate domain adaptation loss and cycle consistency loss for anchor outputs only
        domain_logits = self.domain_classifier(ReverseLayerF.apply(normalized_transformed_vectors, alpha))
        domain_loss = self.criterion(domain_logits, domain_labels)
        cycle_loss = self.cycle_criterion(anchor_outputs, normalized_cycle_vectors)

        return loss, domain_loss, cycle_loss, anchor_outputs

    # def forward(self, input_ids, positive_input_ids, negative_input_ids, labels, negative_labels, domain_labels, alpha):
    #     batch_size, _ = input_ids.size()
    #
    #     # Concatenating input IDs for anchor, positive, and negative examples
    #     concatenated_input_ids = torch.cat((input_ids, positive_input_ids, negative_input_ids), 0)
    #
    #     # Obtain the output embeddings from the encoder
    #     encoder_outputs = self.encoder(concatenated_input_ids, attention_mask=concatenated_input_ids.ne(1))[1]
    #
    #     # Split the outputs back into separate batches
    #     anchor_outputs, positive_outputs, negative_outputs = encoder_outputs.split(batch_size, 0)
    #
    #     # Calculate similarity scores for contrastive loss
    #     positive_similarity = (anchor_outputs * positive_outputs).sum(-1)
    #     negative_similarity = (anchor_outputs * negative_outputs).sum(-1)
    #
    #     # Prepare for softmax calculation
    #     concatenated_anchor_outputs = torch.cat((anchor_outputs, positive_outputs), 0)
    #     concatenated_labels = torch.cat((labels, labels), 0)
    #     similarity_matrix = torch.mm(anchor_outputs, concatenated_anchor_outputs.t())
    #
    #     # Mask to nullify impact of same-element and positive pairs in softmax
    #     mask = labels[:, None] == concatenated_labels[None, :]
    #     similarity_matrix = similarity_matrix * (1 - mask.float()) - 1e9 * mask.float()
    #
    #     # Calculate contrastive loss
    #     probabilities = torch.softmax(torch.cat((positive_similarity[:, None], negative_similarity[:, None], similarity_matrix), -1), -1)
    #     loss = -torch.log(probabilities[:, 0] + 1e-10).mean()
    #
    #     # Transform and cycle-transform for each language pair, avoiding explicit loops
    #     all_transformed_vectors = torch.zeros_like(encoder_outputs)
    #     all_cycle_vectors = torch.zeros_like(encoder_outputs)
    #
    #     for lang_id, lang in enumerate(self.languages):
    #         lang_indices = (domain_labels == lang_id).nonzero(as_tuple=True)[0]
    #         for target_lang_id, target_lang in enumerate(self.languages):
    #             if lang != target_lang:
    #                 # Apply transformation
    #                 transformed_key = f"{lang}_to_{target_lang}"
    #                 transformed_vectors = self.transformations[transformed_key](encoder_outputs[lang_indices])
    #
    #                 # Apply cycle transformation
    #                 cycle_transformed_key = f"{target_lang}_to_{lang}"
    #                 cycle_transformed_vectors = self.transformations[cycle_transformed_key](transformed_vectors)
    #
    #                 # Assign vectors to corresponding positions
    #                 all_transformed_vectors[lang_indices] = transformed_vectors
    #                 all_cycle_vectors[lang_indices] = cycle_transformed_vectors
    #
    #     print(f"all_transformed_vectors shape: {all_transformed_vectors.shape}")  # Add this line
    #     # Normalize vectors
    #     normalized_transformed_vectors = torch.nn.functional.normalize(all_transformed_vectors, p=2, dim=-1)
    #     normalized_cycle_vectors = torch.nn.functional.normalize(all_cycle_vectors, p=2, dim=-1)
    #
    #     print(f"normalized_transformed_vectors shape: {normalized_transformed_vectors.shape}")  # Add this line
    #
    #     # Domain adaptation and cycle consistency loss calculation
    #     domain_logits = self.domain_classifier(ReverseLayerF.apply(normalized_transformed_vectors, alpha))
    #     print(f"domain_logits shape: {domain_logits.shape}")  # Add this line
    #     print(f"domain_labels shape: {domain_labels.shape}")  # Add this line
    #     domain_loss = self.criterion(domain_logits.view(-1, 2), domain_labels.view(-1))
    #     cycle_loss = self.cycle_criterion(encoder_outputs, normalized_cycle_vectors)
    #
    #     return loss, domain_loss, cycle_loss, anchor_outputs

    # def forward(self, input_ids=None, p_input_ids=None, n_input_ids=None, labels=None, negative_labels=None, domain_labels=None, alpha=None):
    #     bs, _ = input_ids.size()
    #     input_ids = torch.cat((input_ids, p_input_ids, n_input_ids), 0)
    #     all_labels = torch.cat((labels, labels, negative_labels), 0)
    #     outputss = self.encoder(input_ids, attention_mask=input_ids.ne(1))[1]
    #     outputs = outputss.split(bs, 0)
    #
    #     prob_1 = (outputs[0] * outputs[1]).sum(-1)  # [batch]
    #     prob_2 = (outputs[0] * outputs[2]).sum(-1)
    #     temp = torch.cat((outputs[0], outputs[1]), 0)
    #     temp_labels = torch.cat((labels, labels), 0)
    #     prob_3 = torch.mm(outputs[0], temp.t())
    #     mask = labels[:, None] == temp_labels[None, :]
    #     prob_3 = prob_3 * (1 - mask.float()) - 1e9 * mask.float()
    #
    #     prob = torch.softmax(torch.cat((prob_1[:, None], prob_2[:, None], prob_3), -1), -1)
    #     loss = torch.log(prob[:, 0] + 1e-10)
    #     loss = -loss.mean()
    #
    #     domain_labelss = torch.cat((domain_labels, domain_labels, domain_labels), 0)
    #
    #     python_indices = (domain_labels == 1).nonzero(as_tuple=True)[0]
    #     java_indices = (domain_labels == 0).nonzero(as_tuple=True)[0]
    #
    #     # Apply transformations in a batched way instead of a loop
    #     forward_vector_python = self.python_java(outputss[python_indices])
    #     forward_vector_java = self.java_python(outputss[java_indices])
    #
    #     # Similarly for cycle vectors
    #     cycle_vector_python = self.java_python(self.python_java(outputss[python_indices]))
    #     cycle_vector_java = self.python_java(self.java_python(outputss[java_indices]))
    #
    #     # Reassemble the vectors based on the original domain ordering
    #     forward_vector = torch.zeros_like(outputss)
    #     forward_vector[python_indices] = forward_vector_python
    #     forward_vector[java_indices] = forward_vector_java
    #
    #     cycle_vector = torch.zeros_like(outputss)
    #     cycle_vector[python_indices] = cycle_vector_python
    #     cycle_vector[java_indices] = cycle_vector_java
    #
    #     forward_vector = torch.true_divide(forward_vector, (torch.norm(forward_vector, dim=-1, keepdim=True) + 1e-13))  # [batch, hidden_size]
    #     cycle_vector = torch.true_divide(cycle_vector, (torch.norm(cycle_vector, dim=-1, keepdim=True) + 1e-13))  # [batch, hidden_size]
    #
    #     all_domain_labelss = torch.cat((domain_labelss, 1 - domain_labelss, domain_labelss), 0)
    #     all_vectors = torch.cat((outputss, forward_vector, cycle_vector), 0)
    #     reversed_pooled_output = ReverseLayerF.apply(all_vectors, alpha)
    #
    #     domain_logits = self.domain_classifier(reversed_pooled_output)
    #     domain_loss = self.criterion(domain_logits.contiguous().view(-1, 2).cuda(), all_domain_labelss.contiguous().view(-1).cuda()).cuda()
    #
    #     cycle_loss = self.cycle_criterion(outputss, cycle_vector)
    #
    #     return loss, domain_loss, cycle_loss, outputs[0]
