# This project uses the structure of MUSE (https://github.com/facebookresearch/MUSE)


from logging import getLogger
from copy import deepcopy
import numpy as np
import torch
from torch import Tensor as torch_tensor
import io
import os
import sys
try:
    import faiss
    FAISS_AVAILABLE = True
    if not hasattr(faiss, 'StandardGpuResources'):
        sys.stderr.write("Impossible to import Faiss-GPU. "
                         "Switching to FAISS-CPU, "
                         "this will be slower.\n\n")

except ImportError:
    sys.stderr.write("Impossible to import Faiss library!! "
                     "Switching to standard nearest neighbors search implementation, "
                     "this will be significantly slower.\n\n")
    FAISS_AVAILABLE = False

from ..dico_builder import get_candidates, build_dictionary
DIC_EVAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'crosslingual', 'dictionaries')
logger = getLogger()

class Evaluator(object):

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.src_emb = trainer.src_emb
        self.tgt_emb = trainer.tgt_emb
        self.src_dico = trainer.src_dico
        self.tgt_dico = trainer.tgt_dico
        self.mapping_G = trainer.mapping_G
        self.mapping_F = trainer.mapping_F
        self.discriminator_A = trainer.discriminator_A
        self.discriminator_B = trainer.discriminator_B
        #self.VAE_A = trainer.VAE_A
        #self.VAE_B = trainer.VAE_B
        self.encoder_A = trainer.encoder_A
        self.decoder_A = trainer.decoder_A
        self.encoder_B = trainer.encoder_B
        self.decoder_B = trainer.decoder_B
        self.params = trainer.params
    
    def dist_mean_cosine(self, to_log, src_to_tgt):
        """
        Mean-cosine model selection criterion.
        """
        if src_to_tgt: 
            
            # get normalized embeddings
            src_emb = self.mapping_G(self.encoder_A(self.src_emb.weight.data)).data
            tgt_emb = self.encoder_B(self.tgt_emb.weight.data).data
            src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
            tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

            # build dictionary
            for dico_method in ['csls_knn_10']:
                dico_build = 'S2T'
                dico_max_size = 10000
                # temp params / dictionary generation
                _params = deepcopy(self.params)
                _params.dico_method = dico_method
                _params.dico_build = dico_build
                _params.dico_threshold = 0
                _params.dico_max_rank = 10000
                _params.dico_min_size = 0
                _params.dico_max_size = dico_max_size
                s2t_candidates = get_candidates(src_emb, tgt_emb, _params) 
                t2s_candidates = get_candidates(tgt_emb, src_emb, _params)
                dico = build_dictionary(src_emb, tgt_emb, _params, s2t_candidates, t2s_candidates)
                # mean cosine
                if dico is None:
                    mean_cosine = -1e9
                else:
                    mean_cosine = (src_emb[dico[:dico_max_size, 0]] * tgt_emb[dico[:dico_max_size, 1]]).sum(1).mean()
                mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch_tensor) else mean_cosine
                logger.info("Mean cosine A->B (%s method, %s build, %i max size): %.5f"
                            % (dico_method, _params.dico_build, dico_max_size, mean_cosine))
                to_log['mean_cosine-%s-%s-%i' % (dico_method, _params.dico_build, dico_max_size)] = mean_cosine


        else:
            
            # get normalized embeddings
            src_emb = self.encoder_A(self.src_emb.weight.data).data
            tgt_emb = self.mapping_F(self.encoder_B(self.tgt_emb.weight.data)).data
            src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
            tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)

            # build dictionary
            for dico_method in ['csls_knn_10']:
                dico_build = 'S2T' ## No need to change here, handled by changing in next piece of code
                dico_max_size = 10000
                # temp params / dictionary generation
                _params = deepcopy(self.params)
                _params.dico_method = dico_method
                _params.dico_build = dico_build
                _params.dico_threshold = 0
                _params.dico_max_rank = 10000
                _params.dico_min_size = 0
                _params.dico_max_size = dico_max_size
                s2t_candidates = get_candidates(src_emb, tgt_emb, _params) 
                t2s_candidates = get_candidates(tgt_emb, src_emb, _params)
                dico = build_dictionary(tgt_emb, src_emb, _params, t2s_candidates, s2t_candidates)
                # mean cosine
                if dico is None:
                    mean_cosine = -1e9
                else:
                    mean_cosine = (tgt_emb[dico[:dico_max_size, 0]] * src_emb[dico[:dico_max_size, 1]]).sum(1).mean()
                mean_cosine = mean_cosine.item() if isinstance(mean_cosine, torch_tensor) else mean_cosine
                logger.info("Mean cosine B->A (%s method, %s build, %i max size): %.5f"
                            % (dico_method, _params.dico_build, dico_max_size, mean_cosine))
                to_log['mean_cosine-%s-%s-%i' % (dico_method, 'T2S', dico_max_size)] = mean_cosine
    
    
    def model_selection_criterion(self, to_log):
        """
        Run Mean-cosine model selection criterion for A->B and B->A
        """
        self.dist_mean_cosine(to_log, src_to_tgt=True)
        self.dist_mean_cosine(to_log, src_to_tgt=False)

    def load_dictionary(self,path, word2id1, word2id2):
        """
        Return a torch tensor of size (n, 2) where n is the size of the
        loader dictionary, and sort it by source word frequency.
        """
        assert os.path.isfile(path)

        pairs = []
        not_found = 0
        not_found1 = 0
        not_found2 = 0

        with io.open(path, 'r', encoding='utf-8') as f:
            for _, line in enumerate(f):
                assert line == line.lower()
                word1, word2 = line.rstrip().split()
                if word1 in word2id1 and word2 in word2id2:
                    pairs.append((word1, word2))
                else:
                    not_found += 1
                    not_found1 += int(word1 not in word2id1)
                    not_found2 += int(word2 not in word2id2)

        logger.info("Found %i pairs of words in the dictionary (%i unique). "
                    "%i other pairs contained at least one unknown word "
                    "(%i in lang1, %i in lang2)"
                    % (len(pairs), len(set([x for x, _ in pairs])),
                       not_found, not_found1, not_found2))

        # sort the dictionary by source word frequencies
        pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
        dico = torch.LongTensor(len(pairs), 2)
        for i, (word1, word2) in enumerate(pairs):
            dico[i, 0] = word2id1[word1]
            dico[i, 1] = word2id2[word2]

        return dico

    def get_nn_avg_dist(self,emb, query, knn):
        """
        Compute the average distance of the `knn` nearest neighbors
        for a given set of embeddings and queries.
        Use Faiss if available.
        """
        if FAISS_AVAILABLE:
            emb = emb.cpu().numpy()
            query = query.cpu().numpy()
            if hasattr(faiss, 'StandardGpuResources'):
                # gpu mode
                res = faiss.StandardGpuResources()
                config = faiss.GpuIndexFlatConfig()
                config.device = 0
                index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
            else:
                # cpu mode
                index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            distances, _ = index.search(query, knn)
            return distances.mean(1)
        else:
            bs = 1024
            all_distances = []
            emb = emb.transpose(0, 1).contiguous()
            for i in range(0, query.shape[0], bs):
                distances = query[i:i + bs].mm(emb)
                best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
                all_distances.append(best_distances.mean(1).cpu())
            all_distances = torch.cat(all_distances)
            return all_distances.numpy()
    def get_word_translation_accuracy(self,lang1, word2id1, emb1, lang2, word2id2, emb2,method):
        """
        Given source and target word embeddings, and a dictionary,
        evaluate the translation accuracy using the precision@k.
        """
        #method="nn"
        path = os.path.join(DIC_EVAL_PATH, '%s-%s.5000-6500.txt' % (lang1, lang2))
            # path = os.path.join(DIC_EVAL_PATH, '%s-%s.0-5000.txt' % (lang1, lang2))
        _dico = self.load_dictionary(path, word2id1, word2id2)
        _dico = _dico.cuda() if emb1.is_cuda else _dico

        # trim dico
        dico = _dico  # [:int(len(_dico)/2)]

        assert dico[:, 0].max() < emb1.size(0)
        assert dico[:, 1].max() < emb2.size(0)

        # normalize word embeddings
        emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
        emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

        # nearest neighbors
        if method == 'nn':
            query = emb1[dico[:, 0]]
            scores = query.mm(emb2.transpose(0, 1))


        # inverted softmax
        elif method.startswith('invsm_beta_'):
            beta = float(method[len('invsm_beta_'):])
            bs = 128
            word_scores = []
            for i in range(0, emb2.size(0), bs):
                scores = emb1.mm(emb2[i:i + bs].transpose(0, 1))
                scores.mul_(beta).exp_()
                scores.div_(scores.sum(0, keepdim=True).expand_as(scores))
                word_scores.append(scores.index_select(0, dico[:, 0]))
            scores = torch.cat(word_scores, 1)

        # contextual dissimilarity measure
        elif method.startswith('csls_knn_'):
            # average distances to k nearest neighbors
            knn = method[len('csls_knn_'):]
            assert knn.isdigit()
            knn = int(knn)
            average_dist1 = self.get_nn_avg_dist(emb2, emb1, knn)
            average_dist2 = self.get_nn_avg_dist(emb1, emb2, knn)
            average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
            average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
            # queries / scores
            query = emb1[dico[:, 0]]
            scores = query.mm(emb2.transpose(0, 1))
            scores.mul_(2)
            scores.sub_(average_dist1[dico[:, 0]][:, None])
            scores.sub_(average_dist2[None, :])

        else:
            raise Exception('Unknown method: "%s"' % method)

        results = []
        top_matches = scores.topk(10, 1, True)[1]
        for k in [1, 5, 10]:
            top_k_matches = top_matches[:, :k]
            _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1).cpu().numpy()
            # allow for multiple possible translations
            matching = {}
            for i, src_id in enumerate(dico[:, 0].cpu().numpy()):
                # evaluate precision@k
                matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
            precision_at_k = 100 * np.mean(list(matching.values()))
            logger.info("%i source words - %s - Precision at k = %i: %f" %
                        (len(matching), method, k, precision_at_k))
            results.append(('precision_at_%i' % k, precision_at_k))

        return results

    def word_translation(self, to_log):
        """
        Evaluation on word translation.
        """
        # mapped word embeddings

        src_emb = self.mapping_G(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data

        for method in ['csls_knn_10']:
            print(method)
            results = self.get_word_translation_accuracy(
                self.src_dico.lang, self.src_dico.word2id, src_emb,
                self.tgt_dico.lang, self.tgt_dico.word2id, tgt_emb,
                method=method,
                dico_eval=self.params.dico_eval)
            to_log.update([('%s-%s' % (k, method), v) for k, v in results])
    def all_eval(self, to_log):
        """
        Run all evaluations.
        """

        src_emb = self.mapping_G(self.encoder_A(self.src_emb.weight.data)).data
        tgt_emb = self.encoder_B(self.tgt_emb.weight.data).data
        for method in ["nn",'csls_knn_10']:
            self.get_word_translation_accuracy(self.src_dico.lang, self.src_dico.word2id, src_emb,self.tgt_dico.lang, self.tgt_dico.word2id, tgt_emb,method=method)
       # self.word_translation(to_log)

        src_emb = self.mapping_F(self.encoder_B(self.tgt_emb.weight.data)).data
        tgt_emb = self.encoder_A(self.src_emb.weight.data).data
        for method in ["nn", 'csls_knn_10']:
            self.get_word_translation_accuracy(self.tgt_dico.lang, self.tgt_dico.word2id, src_emb, self.src_dico.lang,
                                           self.src_dico.word2id, tgt_emb, method=method)

    