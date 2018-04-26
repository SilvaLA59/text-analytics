# -*- coding: utf-8 -*-
import nltk
import requests
import json
import re
from asyncio.tasks import sleep
from rake_nltk import Rake
from flask import Flask, jsonify,request

# <-- Contrução da base de dados

base = [('você é tão irritante', 'Annoying'),
        ('como você é irritante', 'Annoying'),
        ('você está me irritando', 'Annoying'),
        ('eu acho você irritante', 'Annoying'),
        ('você me irrita', 'Annoying'),
        ('você está me irritando tanto', 'Annoying'),
        ('você é incrivelmente irritante', 'Annoying'),
        ('você é irritante', 'Annoying'),
        ('você é ruim mesmo', 'Bad'),
        ('você não está me ajudando', 'Bad'),
        ('você é ruim', 'Bad'),
        ('você é muito ruim', 'Bad'),
        ('você é chato', 'Bad'),
        ('você não serve para nada', 'Bad'),
        ('você é horrível', 'Bad'),
        ('você é inútil', 'Bad'),
        ('você é uma perda de tempo', 'Bad'),
        ('você é nojento', 'Bad'),
        ('você é incrível', 'Good'),
        ('você é boa', 'Good'),
        ('você é a melhor', 'Good'),
        ('você trabalha bem', 'Good'),
        ('Gosto muito do seu trabalho', 'Good'),
        ('você é muito boa nisso', 'Good'),
        ('você é uma verdadeira profissional', 'Good'),
        ('você é boa nisso', 'Good'),
        ('você me ajuda muito', 'Good'),
        ('você é profissa', 'Good'),
        ('você é uma profissional', 'Good'),
        ('opa', 'Hello'),
        ('olá', 'Hello'),
        ('eae', 'Hello'),
        ('fala aí', 'Hello'),
        ('oi', 'Hello'),
        ('saudações', 'Hello'),
        ('oi tudo bem', 'Hello'),
        ('há quanto tempo', 'Hello'),
        ('fala', 'Hello'),
        ('e aí', 'Hello')
]
# stopwords = []
# stopwords = ['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'uma', 'os', 'no', 'é',
#                 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'à', 'seu',
#                 'sua', 'ou', 'quando', 'muito', 'nos', 'já', 'também', 'só', 'pelo', 'pela', 'até', 'eu',
#                 'isso', 'ela', 'entre', 'depois', 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse',
#                 'eles', 'você', 'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'numa', 'pelos', 'elas', 'qual',
#                 'nós', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'dele', 'tu', 'te', 'vocês', 'vos', 'lhes',
#                 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas',
#                 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo', 'estou', 'está',
#                 'estamos', 'estão', 'estive', 'esteve', 'estivemos', 'estiveram', 'estava', 'estávamos', 'estavam', 'estivera',
#                 'estivéramos', 'esteja', 'estejamos', 'estejam', 'estivesse', 'estivéssemos', 'estivessem', 'estiver', 'estivermos',
#                 'estiverem', 'hei', 'há', 'havemos', 'hão', 'houve', 'houvemos', 'houveram', 'houvera', 'houvéramos', 'haja', 'hajamos',
#                 'hajam', 'houvesse', 'houvéssemos', 'houvessem', 'houver', 'houvermos', 'houverem', 'houverei', 'houverá', 'houveremos',
#                 'houverão', 'houveria', 'houveríamos', 'houveriam', 'sou', 'somos', 'são', 'era', 'éramos','eram', 'fui', 'foi', 'fomos',
#                 'foram', 'fora', 'fôramos', 'seja', 'sejamos', 'sejam', 'fosse', 'fôssemos', 'fossem', 'for', 'formos', 'forem', 'serei',
#                 'será', 'seremos', 'serão', 'seria', 'seríamos', 'seriam', 'tenho', 'tem', 'temos', 'tém', 'tinha', 'tínhamos', 'tinham',
#                 'tive', 'teve', 'tivemos', 'tiveram', 'tivera', 'tivéramos', 'tenha', 'tenhamos', 'tenham', 'tivesse', 'tivéssemos', 'tivessem',
#                 'tiver', 'tivermos', 'tiverem', 'terei', 'terá', 'teremos', 'terão', 'teria', 'teríamos', 'teriam']

stopwords = nltk.corpus.stopwords.words('portuguese')
# print(stopwords)

target_sentences = [ 'você é incrível',
        'você é boa',
        'você é a melhor',
        'você trabalha bem',
        'você é muito boa nisso',
        'você é uma verdadeira profissional',
        'você é boa nisso',
        'você me ajuda muito',
        'você é profissa',
        'você é uma profissional']

various_sentences = [ 'você é tão irritante',
        'como você é irritante',
        'você está me irritando',
        'eu acho você irritante',
        'você me irrita',
        'você está me irritando tanto',
        'você é incrivelmente irritante',
        'você é irritante',
        'você é ruim mesmo',
        'você não está me ajudando',
        'você é ruim',
        'você é muito ruim',
        'você é chato',
        'você não serve para nada',
        'você é horrível',
        'você é inútil',
        'você é uma perda de tempo',
        'você é nojento' ]


# ------------------------ FUNÇOES ------------------------
# remover as stop words do texto
def removerStopWords(texto):
        frases = []
        for (palavras, emocao) in texto:
                semStopWords = [p for p in palavras.split() if p not in stopwords]
                frases.append((semStopWords, emocao))
        return frases

# Identificar o radical da palavra com Stemming
def aplicarStemmer(texto):
        stemmer = nltk.stem.RSLPStemmer()
        frasesStemming = []
        for (palavras, emocao) in texto:
                stemmerAplicado = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwords]
                frasesStemming.append((stemmerAplicado, emocao))
        return frasesStemming
# baseStemming = aplicarStemmer(base)
#print(aplicarStemmer(base))
def aplicarSnowballStemmer(texto):
        stemmer = nltk.stem.SnowballStemmer('portuguese')
        fraseStemming = []
        for (palavras) in texto.split():
                stemmerAplicado = [p for p in palavras.split()]
                fraseStemming.append(str(stemmer.stem(stemmerAplicado[0])))
        return fraseStemming
def aplicarRSLPStemmer(texto):
        stemmer = nltk.stem.RSLPStemmer()
        fraseStemming = []
        for (palavras) in texto.split():
                stemmerAplicado = [p for p in palavras.split()]
                fraseStemming.append(str(stemmer.stem(stemmerAplicado[0])))
        return fraseStemming
#mensagemStopWords = aplicarStemmerFrase(mensagem)
# print('Snowball: %s' % aplicarSnowballStemmer('Essa novidade é nova para mim novamente'))
# print('RSLPStemmer: %s' % aplicarRSLPStemmer('Essa novidade é nova para mim novamente'))

#buscar apenas as palavras da base
def buscarPalavras(frases):
        palavrasBase = []
        for (palavras, emocao) in frases:
                palavrasBase.extend(palavras)
        return palavrasBase

# <-- Remover palavras repetidas
def removerPalavrasRepetidas(palavras):
        palavras = nltk.FreqDist(palavras) #Retorna as palavras com a frequencia
        palavrasSemRepeticao = palavras.keys() #Remove repetições
        return palavrasSemRepeticao
# -->

def extrairPalavras(documento):
        doc = set(documento)
        caracteristicas = {}

        palavrasBase = buscarPalavras(baseSemStopWords)        
        palavrasBaseSemRepeticao = removerPalavrasRepetidas(palavrasBase)
        
        for palavras in palavrasBaseSemRepeticao:
                caracteristicas['%s' % palavras] = (palavras in doc)
        return caracteristicas

def removerCaracteresMap(charMap, texto):
        textoRetorno = texto
        for i in range(0, len(charMap)):
                textoRetorno = textoRetorno.replace(charMap[i], "")
        return textoRetorno

# def extrairPalavrasStemming(documento):
#         doc = set(documento)
#         caracteristicas = {}
#         for palavras in palavrasBaseSemRepeticaoStemming:
#                 caracteristicas['%s' % palavras] = (palavras in doc)
#         return caracteristicas

def getKeyWords(frase):
        # r = Rake(stopwords=stopwords, language='portuguese')
        r = Rake(stopwords=stopwords, language='portuguese')
        r.extract_keywords_from_text(frase)
        # r.get_ranked_phrases()
        # print(r.get_ranked_phrases_with_scores())
        retorno = r.get_ranked_phrases_with_scores()
        retornoJson = []
        for (score, keyword) in retorno:
                #retornoJson.append({ "keyword": keyword, "score": score })
                retornoJson.append(keyword)
        
        return retornoJson
#getKeyWords('MINHA VIDA ESTA MUDANDO MUITO DESDE QUE ESTOU VENDO ESSAS LIVE WENDELL PARABÉNS IMPARAVEL!!!!')

def textAnalize(texto):
        #frase = texto.lower()
        caracteresRemoverMap = r'[+-./?!,"\']'
        texto = re.sub(caracteresRemoverMap,"", texto)
        texto = texto.lower()
        texto = re.sub(r'[àáâã]',"a", texto)
        texto = re.sub(r'[èéê]',"e", texto)
        texto = re.sub(r'[ìíî]',"i", texto)
        texto = re.sub(r'[òóôõ]',"o", texto)
        texto = re.sub(r'[ùúû]',"u", texto)
        texto = re.sub(r'[ç]',"c", texto)
        
        msgPlanilhada = extrairPalavras(texto.split())
        distribuicao = classificador.prob_classify(msgPlanilhada)
        # print('msgPlhanilhada %s' % msgPlanilhada)
        retornoDist = {}

        for classe in distribuicao.samples():
                retornoDist['%s' % classe] = round(distribuicao.prob(classe), 2)

        retorno = classificador.classify(msgPlanilhada)
        
        # Tratar se a acuracidade da inteção menor que 70%, responder com inteção 'none'
        if retornoDist['%s' % retorno] < 0.25:
                retorno = 'none'

        return retorno

def positividade(texto):
        #from nltk.classify import PositiveNaiveBayesClassifier
        positive_featuresets = list(map(features, target_sentences))
        unlabeled_featuresets = list(map(features, various_sentences))
        classifier = nltk.classify.PositiveNaiveBayesClassifier.train(positive_featuresets, unlabeled_featuresets)

        distribuicao = classifier.prob_classify(features(texto))
        retornoDist = {}

        for classe in distribuicao.samples():
                retornoDist['%s' % classe] = round(distribuicao.prob(classe), 2)

        retorno = classifier.classify(features(texto))
                
        if retorno:
                score = retornoDist['%s' % retorno]
        else:
                score = 1 - retornoDist['%s' % retorno]
        
        return jsonify({ "Positive": retorno, "Score": score })

def features(sentence):
        words = sentence.lower().split()
        return dict(('contains(%s)' % w, True) for w in words)
# ------------------------ FUNÇOES ------------------------

# ------------------------ CHAMADAS ------------------------

try:         
        baseSemStopWords = removerStopWords(base)
        # print(baseSemStopWords)        
        palavrasBase = buscarPalavras(baseSemStopWords)
        # palavrasBaseStemming = buscarPalavras(baseStemming)
        #print(palavrasBase)        
        palavrasBaseSemRepeticao = removerPalavrasRepetidas(palavrasBase)
        # palavrasBaseSemRepeticaoStemming = removerPalavrasRepetidas(palavrasBaseStemming)
        #print(palavrasBaseSemRepeticao)

        baseCompleta = nltk.classify.apply_features(extrairPalavras, baseSemStopWords)
        # baseCompletaStemming = nltk.classify.apply_features(extrairPalavrasStemming, baseStemming)
        #print(baseCompleta[0])

# Contrução da base de dados -->

        #Montar a tabela de probabilidades(treinamento) da base construida
        classificador = nltk.NaiveBayesClassifier.train(baseCompleta)
        # classificadorStemming = nltk.NaiveBayesClassifier.train(baseCompletaStemming)
        #print(classificador.labels())
        #print(classificador.show_most_informative_features(5))

except ValueError:
        print('Erro de valor')
except:
        print("Unexpected error")
        raise

# ------------------------ CHAMADAS ------------------------

#API
app = Flask(__name__)

@app.route("/intent", methods=['POST'])
def getIntet():
        dados = request.json
        mensagem = dados['texto']

        # <-- Testes
        # frasesTeste = nltk.tokenize.sent_tokenize(mensagem) # Separa um texto grande em frases
        # for item in range(0, len(frasesTeste)):
        #         print(frasesTeste[item])
        # tokens = nltk.word_tokenize(mensagem, 'portuguese') # Separa as palavras da frase em tokens
        # classes = nltk.pos_tag(tokens, lang='portuguese') # Identifica qual a classe gramatical de cada palavra da frase
        # print(classes)

        # Testes --> 
        
        # Remover caracteres de caracteresRemoverMap no texto recebido        
        caracteresRemoverMap = r'[+-./?!,"\']'
        mensagem = mensagem.lower()
        mensagem = re.sub(caracteresRemoverMap,"", mensagem)        
        mensagem = re.sub(r'[àáâã]',"a", mensagem)
        mensagem = re.sub(r'[èéê]',"e", mensagem)
        mensagem = re.sub(r'[ìíî]',"i", mensagem)
        mensagem = re.sub(r'[òóôõ]',"o", mensagem)
        mensagem = re.sub(r'[ùúû]',"u", mensagem)
        mensagem = re.sub(r'[ç]',"c", mensagem)
        
        msgPlanilhada = extrairPalavras(mensagem.split())
        
        distribuicao = classificador.prob_classify(msgPlanilhada)
        retornoDist = {}

        for classe in distribuicao.samples():
                retornoDist['%s' % classe] = round(distribuicao.prob(classe), 2)

        retorno = classificador.classify(msgPlanilhada)
        # print(classificador.classify(msgPlanilhada))

        # <-- Analise por Stemming(Radiacal)
        # msgStemming = aplicarStemmerFrase(mensagem)
        # msgPlanilhadaStemming = extrairPalavras(msgStemming)

        # distribuicaoStemming = classificadorStemming.prob_classify(msgPlanilhadaStemming)
        # retornoDistStemming = {}

        # for classe in distribuicaoStemming.samples():
        #         retornoDistStemming['%s' % classe] = round(distribuicaoStemming.prob(classe), 2)
        #         print("%s: %f" % (classe, distribuicaoStemming.prob(classe)))

        # retornoStemming = classificadorStemming.classify(msgPlanilhadaStemming)
        # Analise por Stemming(Radiacal) -->
        
        # Tratar se a acuracidade da inteção do menor que 80%, responder com inteção 'none'
        if retornoDist['%s' % retorno] < 0.25:
                retorno = 'none'
        #import time
        #time.sleep(3)
        return jsonify( 
        { 
                'mensagem': dados['texto'],
                'intencao': retorno,
                'distribuicao': retornoDist 
        })

@app.route("/keywords", methods=['POST'])
def keywordsAPI():
        dados = request.json
        frase = dados['texto']
        caracteresRemoverMap = r'[+-./?!,"\']'
        frase = re.sub(caracteresRemoverMap,"", frase)
        # frase = re.sub('...'," ", frase)
        # frase = frase.lower()

        retorno = getKeyWords(frase)
        return jsonify({'texto': frase, 'keyWords': retorno})

@app.route("/textanalises", methods=['POST'])
def textAnalizeAPI():
        dados = request.json
        texto = dados['texto']
        
        keywords = getKeyWords(texto)
        intent = textAnalize(texto)
        sentiment = positividade(texto)
        positive = json.loads(sentiment.response[0])
        
        return jsonify({
                "intent": intent,
                "keywords": keywords,
                "sentiment": positive['Score'],
                "texto": dados['texto']
        })


if __name__ == '__main__':
        app.run()

