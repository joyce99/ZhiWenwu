# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import copy

import embeddings
from cupy_utils import *
import time
import argparse
import collections
import numpy as np
import sys
import datetime


BATCH_SIZE = 2#500


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embeddings of two languages in a shared space in word translation induction')
    parser.add_argument('src_embeddings', help='the source language embeddings')
    parser.add_argument('trg_embeddings', help='the target language embeddings')
    parser.add_argument('src_definition', help='the source language embeddings')
    parser.add_argument('trg_definition', help='the target language embeddings')
    parser.add_argument('mid_file', help='the mid file')
    parser.add_argument('result_file', help='the result file')
    parser.add_argument('-d', '--dictionary', default=sys.stdin.fileno(), help='the test dictionary file (defaults to stdin)')
    parser.add_argument('--retrieval', default='csls', choices=['nn', 'invnn', 'invsoftmax', 'csls','nn1','csls1'], help='the retrieval method (nn: standard nearest neighbor; invnn: inverted nearest neighbor; invsoftmax: inverted softmax; csls: cross-domain similarity local scaling)')
    parser.add_argument('--inv_temperature', default=1, type=float, help='the inverse temperature (only compatible with inverted softmax)')
    parser.add_argument('--inv_sample', default=None, type=int, help='use a random subset of the source vocabulary for the inverse computations (only compatible with inverted softmax)')
    parser.add_argument('-k', '--neighborhood', default=10, type=int, help='the neighborhood size (only compatible with csls)')
    parser.add_argument('--dot', action='store_true', help='use the dot product in the similarity computations instead of the cosine')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--seed', type=int, default=0, help='the random seed')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    args = parser.parse_args()

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # Read input embeddings and words
    srcfile = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)
    # NumPy/CuPy management
    if args.cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        x = xp.asarray(x)
        z = xp.asarray(z)
    else:
        xp = np
    xp.random.seed(args.seed)

    # Length normalize embeddings so their dot product effectively computes the cosine similarity
    if not args.dot:
        embeddings.length_normalize(x)
        embeddings.length_normalize(z)

    # Build word to index map,将原来的列表变成键值形式<key,value>，<word,num>'chickweed': 104501, 'bressingham': 104502, 'lonicera': 104503,
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    #'chickweed': 104501, 'bressingham': 104502, 'lonicera': 104503,
    #print("src_word2ind",src_word2ind)
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # Read dictionary and compute coverage
    f = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')

    #src2trg为对应翻译的索引，{0: {10, 3, 4, 13}, 4: {8}, 16: {266, 139, 39}, 10: {17}, 12: {136, 6}}
    #0为the词的索引，
    # the el
    # the las
    # the los
    # the la
    src2trg = collections.defaultdict(set)
    oov = set()
    vocab = set()
    for line in f:
        #这是原来的Vecmap的字典函数
        src, trg = line.split()
        try:
            #src_ind为该词在src_word2ind中的索引，即为值,键为词
            src_ind = src_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src2trg[src_ind].add(trg_ind)
            vocab.add(src)
        except KeyError:
            oov.add(src)
    #print(src2trg)
    src = list(src2trg.keys())
    oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
    coverage = len(src2trg) / (len(src2trg) + len(oov))
    print("覆盖率=============")
    # Find translations
    translation = collections.defaultdict(int)
    if args.retrieval == 'nn':  # Standard nearest neighbor
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            #print(src[i:j])
            #src为包含源词的索引，因此src[i]为src中第i个词的索引，x[src[i]]为第i个词的词向量,1*300,x[src[i:j]]表示第i个词到第j个词的词向量
            #x[src[i:j]]为500*300，z.T为300*200000，点积后为500*200000，即银行中最大值即为翻译结果
            similarities = x[src[i:j]].dot(z.T)
            #argmax为最大元素索引，axis=1表示为列的最大索引,即本文求top1
            nn = similarities.argmax(axis=1).tolist()
            #print(nn)
            #print(max(nn))
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
                #通过值求该值对应的键
                #[x for x, v in src_word2ind.items() if v == src[i+k]]为源词
                #[x for x, v in trg_word2ind.items() if v == nn[k]]为得到的翻译词
                #print([x for x, v in src_word2ind.items() if v == src[i+k]])
                #print([x for x, v in trg_word2ind.items() if v == nn[k]])
        #print(translation) translation =  {0: 11, 4: 8, 16: 39, 10: 5, 12: 6, 15: 16, 26: 5607, 25: 41,...}
    # 此处为获取top@10的翻译词
    elif args.retrieval == 'nn1':  # Standard nearest neighbor
        fout = open(args.mid_file, "w", encoding='utf-8')
        time = 0
        print("翻译词总共有：",len(src),"个")
        for i in range(0, len(src)):
            #print(src[i:j])
            #src为包含源词的索引，因此src[i]为src中第i个词的索引，x[src[i]]为第i个词的词向量,1*300,x[src[i:j]]表示第i个词到第j个词的词向量
            #x[src[i:j]]为500*300，z.T为300*200000，点积后为500*200000，即银行中最大值即为翻译结果
            similarities = x[src[i]].dot(z.T)
            #argmax为最大元素索引，axis=1表示为列的最大索引,即本文求top1
            source_word = [x for x, v in src_word2ind.items() if v == src[i]]
            traned_words = []
            traned_sim = []
            # print(time)
            time = time + 1
            fout.write(source_word.pop())
            fout.write('\t')
            for j in range(10):
                nn = similarities.argmax(axis=0).tolist()
                sim = copy.copy(similarities[nn])
                similarities[nn] = 0
                traned_words.append([x for x, v in trg_word2ind.items() if v == nn])
                traned_sim.append(sim)
                fout.write(str(traned_words[j].pop()))
                fout.write(" ")
                fout.write(str(traned_sim[j]))
                fout.write(" ")
                # fout.write('('.join(sim).join(')  '))
                #print(traned_words)
            # fout.write('{0}\t{1}\n'.format(source_word.pop(), ' '.join(k.pop() for k in traned_words)))
            # fout.write(source_word.pop())
            fout.write('\n')

        fout.close()
        print("文件写入完毕")
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            #print(src[i:j])
            #src为包含源词的索引，因此src[i]为src中第i个词的索引，x[src[i]]为第i个词的词向量,1*300,x[src[i:j]]表示第i个词到第j个词的词向量
            #x[src[i:j]]为500*300，z.T为300*200000，点积后为500*200000，即银行中最大值即为翻译结果
            similarities = x[src[i:j]].dot(z.T)
            #argmax为最大元素索引，axis=1表示为列的最大索引,即本文求top1
            nn = similarities.argmax(axis=1).tolist()
            # print(len(nn))
            # print(nn)
            for k in range(j-i):
                translation[src[i+k]] = nn[k]

        #print(translation) translation =  {0: 11, 4: 8, 16: 39, 10: 5, 12: 6, 15: 16, 26: 5607, 25: 41,...}
    elif args.retrieval == 'invnn':  # Inverted nearest neighbor
        best_rank = np.full(len(src), x.shape[0], dtype=int)
        best_sim = np.full(len(src), -100, dtype=dtype)
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            similarities = z[i:j].dot(x.T)
            ind = (-similarities).argsort(axis=1)
            ranks = asnumpy(ind.argsort(axis=1)[:, src])
            sims = asnumpy(similarities[:, src])
            for k in range(i, j):
                for l in range(len(src)):
                    rank = ranks[k-i, l]
                    sim = sims[k-i, l]
                    if rank < best_rank[l] or (rank == best_rank[l] and sim > best_sim[l]):
                        best_rank[l] = rank
                        best_sim[l] = sim
                        translation[src[l]] = k
    elif args.retrieval == 'invsoftmax':  # Inverted softmax
        sample = xp.arange(x.shape[0]) if args.inv_sample is None else xp.random.randint(0, x.shape[0], args.inv_sample)
        partition = xp.zeros(z.shape[0])
        for i in range(0, len(sample), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(sample))
            partition += xp.exp(args.inv_temperature*z.dot(x[sample[i:j]].T)).sum(axis=1)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            p = xp.exp(args.inv_temperature*x[src[i:j]].dot(z.T)) / partition
            nn = p.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
    elif args.retrieval == 'csls1':  # Cross-domain similarity local scaling
        time = 0
        #fout = open("all_en-es.test.txt", "w", encoding='utf-8')
        time = 0
        #print("翻译词总共有：", len(src), "个")
        knn_sim_bwd = xp.zeros(z.shape[0])

        fout = open(args.mid_file, "w", encoding='utf-8')
        time = 0
        print("翻译词总共有：", len(src), "个")
        for i in range(0, len(src)):
            # print(src[i:j])
            # src为包含源词的索引，因此src[i]为src中第i个词的索引，x[src[i]]为第i个词的词向量,1*300,x[src[i:j]]表示第i个词到第j个词的词向量
            # x[src[i:j]]为500*300，z.T为300*200000，点积后为500*200000，即银行中最大值即为翻译结果
            knn_sim_bwd[i] = topk_mean(z[i].dot(x.T), k=10, inplace=True)


            similarities =  2*x[src[i]].dot(z.T) - knn_sim_bwd
            # argmax为最大元素索引，axis=1表示为列的最大索引,即本文求top1
            source_word = [x for x, v in src_word2ind.items() if v == src[i]]
            traned_words = []
            traned_sim = []
            # print(time)
            time = time + 1
            fout.write(source_word.pop())
            fout.write('\t')
            for j in range(10):
                nn = similarities.argmax(axis=0).tolist()
                sim = copy.copy(similarities[nn])
                similarities[nn] = 0
                traned_words.append([x for x, v in trg_word2ind.items() if v == nn])
                traned_sim.append(sim)
                fout.write(str(traned_words[j].pop()))
                fout.write(" ")
                fout.write(str(traned_sim[j]))
                fout.write(" ")
                # fout.write('('.join(sim).join(')  '))
                # print(traned_words)
            # fout.write('{0}\t{1}\n'.format(source_word.pop(), ' '.join(k.pop() for k in traned_words)))
            # fout.write(source_word.pop())
            fout.write('\n')

        fout.close()


        for i in range(0, z.shape[0], BATCH_SIZE):
            print(time)
            time = time + 1
            j = min(i + BATCH_SIZE, z.shape[0])
            knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=args.neighborhood, inplace=True)
        time = 0
        for i in range(0, len(src), BATCH_SIZE):
            print(time)
            time = time + 1
            j = min(i + BATCH_SIZE, len(src))
            similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
            nn = similarities.argmax(axis=1).tolist()
            for k in range(j-i):
                translation[src[i+k]] = nn[k]
    elif args.retrieval == 'nn2':  # Standard nearest neighbor
        fout = open(args.mid_file, "w", encoding='utf-8')
        time = 0
        print("翻译词总共有：",len(src),"个")
        for i in range(0, len(src)):
            #print(src[i:j])
            #src为包含源词的索引，因此src[i]为src中第i个词的索引，x[src[i]]为第i个词的词向量,1*300,x[src[i:j]]表示第i个词到第j个词的词向量
            #x[src[i:j]]为500*300，z.T为300*200000，点积后为500*200000，即银行中最大值即为翻译结果
            similarities = x[src[i]].dot(z.T)
            #argmax为最大元素索引，axis=1表示为列的最大索引,即本文求top1
            source_word = [x for x, v in src_word2ind.items() if v == src[i]]
            traned_words = []
            traned_sim = []
            # print(time)
            time = time + 1
            fout.write(source_word.pop())
            fout.write('\t')
            for j in range(10):
                nn = similarities.argmax(axis=0).tolist()
                sim = copy.copy(similarities[nn])
                similarities[nn] = 0
                traned_words.append([x for x, v in trg_word2ind.items() if v == nn])
                traned_sim.append(sim)
                fout.write(str(traned_words[j].pop()))
                fout.write(" ")
                fout.write(str(traned_sim[j]))
                fout.write(" ")
                # fout.write('('.join(sim).join(')  '))
                #print(traned_words)
            # fout.write('{0}\t{1}\n'.format(source_word.pop(), ' '.join(k.pop() for k in traned_words)))
            # fout.write(source_word.pop())
            fout.write('\n')

        fout.close()

    elif args.retrieval == 'csls':  # Cross-domain similarity local scaling
        print("csls=============")
        knn_sim_bwd = xp.zeros(z.shape[0])
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=args.neighborhood, inplace=True)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = 2 * x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
            nn = similarities.argmax(axis=1).tolist()
            high = similarities[nn]
            print(high)
            for k in range(j - i):
                translation[src[i + k]] = nn[k]

    # Compute accuracy
    #for i in src:
       # print(src2trg[i])

    accuracy = np.mean([1 if translation[i] in src2trg[i] else 0 for i in src])
    print('Coverage:{0:7.2%}  Accuracy:{1:7.2%}'.format(coverage, accuracy))


if __name__ == '__main__':
    start = datetime.datetime.now()
    print(start)
    main()
    end = datetime.datetime.now()
    print(end)
    print((end-start).seconds)
