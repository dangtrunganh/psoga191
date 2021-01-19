import random
from features import compute_fitness
from preprocess import preprocess_raw_sent
from preprocess import sim_with_title
from preprocess import sim_with_doc
from preprocess import sim_2_sent
from preprocess import count_noun
from copy import copy
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import nltk
import os.path
import statistics as sta
import re
from preprocess import preprocess_for_article
from preprocess import preprocess_numberOfNNP
import time
import os
from rouge import Rouge
from shutil import copyfile
import pandas as pd



class Summerizer(object):
    def __init__(self, title, sentences, raw_sentences, population_size, max_generation, crossover_rate, mutation_rate, num_picked_sents, simWithTitle, simWithDoc, sim2sents, number_of_nouns):
        self.title = title
        self.raw_sentences = raw_sentences
        self.sentences = sentences
        self.num_objects = len(sentences)
        self.population_size = population_size
        self.max_generation = max_generation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_picked_sents = num_picked_sents
        self.simWithTitle = simWithTitle
        self.simWithDoc = simWithDoc
        self.sim2sents = sim2sents
        self.number_of_nouns = number_of_nouns


    # def generate_population(self, amount):
#         print("Generating population...")
#         population = []
#         current = 0
#         if self.num_objects % self.num_picked_sents != 0:
#             m = self.num_picked_sents
#         else:
#             m = self.num_picked_sents - 1
#         for i in range(amount):
#             agent = np.zeros(self.num_objects)
#             for j in range(current, current + m):
#                 agent[j%self.num_objects] = 1
#             agent = agent.tolist()
#             #p_best_position
#             pbest_position = agent

#             #p_best_value
#             fitness = compute_fitness(self.title, self.sentences, agent, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns)
            
#             #velocity
#             velocity = np.zeros(self.num_objects)
#             velocity[np.random.choice(list(range(self.num_objects)), self.num_picked_sents, replace=False)] = 1
#             velocity = velocity.tolist()


#             # print("fitness: {:.4f}" , format(fitness))
#             # print(agent)
#             population.append((agent, fitness, pbest_position, velocity))
#             current += m
#         return population 

    def generate_population(self, amount):
        # print("Generating population...")
        population = []
        for i in range(amount):

            #position
            agent = np.zeros(self.num_objects)
            agent[np.random.choice(list(range(self.num_objects)), self.num_picked_sents, replace=False)] = 1
            agent = agent.tolist()
            
            #p_best_position
            pbest_position = agent

            #p_best_value
            fitness = compute_fitness(self.title, self.sentences, agent, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns)
            
            #velocity
            velocity = np.zeros(self.num_objects)
            velocity[np.random.choice(list(range(self.num_objects)), self.num_picked_sents, replace=False)] = 1
            velocity = velocity.tolist()


            # print("fitness: {:.4f}" , format(fitness))
            # print(agent)
            population.append((agent, fitness, pbest_position, velocity))
        return population 


    def words_count(self, sentences):
        words = nltk.word_tokenize(sentences)
        return len(words)


    def sum_of_words(self, individual):
        sum = 0
        agent = individual[0][:]
        for i in range(self.num_objects):
            if agent[i]==1:
                sum += self.words_count(self.sentences[i])
        return sum


    def roulette_select(self, total_fitness, population):
        fitness_slice = np.random.rand() * total_fitness
        fitness_so_far = 0.0
        for phenotype in population:
            fitness_so_far += phenotype[1]
            if fitness_so_far >= fitness_slice:
                return phenotype
        return None


    def rank_select(self, population):
        ps = len(population)
        if ps == 0:
            return None
        population = sorted(population, key=lambda x: x[1], reverse=True)
        fitness_value = []
        for individual in population:
            fitness_value.append(individual[1])

        fittest_individual = max(fitness_value)
        medium_individual = sta.median(fitness_value)
        selective_pressure = fittest_individual - medium_individual
        j_value = 1
        a_value = np.random.rand()   
        for agent in population:
            if ps == 0:
                return None
            elif ps == 1:
                return agent
            else:
                range_value = selective_pressure - (2*(selective_pressure - 1)*(j_value - 1))/( ps - 1) 
                prb = range_value/ps
                if prb > a_value:
                    return agent
            j_value +=1

                
    def crossover(self, individual_1, individual_2, max_sent):
        # check tỷ lệ crossover
        if self.num_objects < 2 or random.random() >= self.crossover_rate:
            return individual_1[:], individual_2[:]
        
        #tìm điểm chéo 1
        crossover_point = 1 + random.randint(0, self.num_objects - 2)
        agent_1a = individual_1[0][:crossover_point] + individual_2[0][crossover_point:]
        fitness_1a = compute_fitness(self.title, self.sentences, agent_1a, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns)
        agent_1b = individual_2[0][:crossover_point] + individual_1[0][crossover_point:]
        fitness_1b = compute_fitness(self.title, self.sentences, agent_1b, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns)

        velocity = np.zeros(self.num_objects)
        velocity = velocity.tolist()

        if fitness_1a > fitness_1b:
            child_1 = (agent_1a, fitness_1a, agent_1a, velocity)
        else:
            child_1 = (agent_1b, fitness_1b, agent_1a, velocity)

        sum_sent_in_summary = sum(child_1[0])
        agent_1 = child_1[0]
        fitness_1 = child_1[1]
        if sum_sent_in_summary > max_sent:
            while(sum_sent_in_summary > max_sent):
                remove_point = 1 + random.randint(0, self.num_objects - 2)
                if agent_1[remove_point] == 1:
                    agent_1[remove_point] = 0
                    sent = self.sentences[remove_point]
                    sum_sent_in_summary -=1            
            fitness_1 = compute_fitness(self.title, self.sentences, agent_1, self.simWithTitle, self.simWithDoc,self.sim2sents, self.number_of_nouns)
            child_1 = (agent_1, fitness_1, agent_1, velocity)


        #tìm điểm chéo 2
        crossover_point_2 = 1 + random.randint(0, self.num_objects - 2)
        
        agent_2a = individual_1[0][:crossover_point_2] + individual_2[0][crossover_point_2:]
        fitness_2a = compute_fitness(self.title, self.sentences, agent_2a, self.simWithTitle, self.simWithDoc,self.sim2sents, self.number_of_nouns)
        agent_2b = individual_2[0][:crossover_point_2] + individual_1[0][crossover_point_2:]
        # agent_2 = individual_2[0][:crossover_point] + individual_1[0][crossover_point:]
        fitness_2b = compute_fitness(self.title, self.sentences, agent_2b, self.simWithTitle, self.simWithDoc,self.sim2sents, self.number_of_nouns)
        if fitness_2a > fitness_2b:
            child_2 = (agent_2a, fitness_2a, agent_2a, velocity)
        else:
            child_2 = (agent_2b, fitness_2b, agent_2b, velocity)        
        
        sum_sent_in_summary_2 = sum(child_2[0])
        agent_2 = child_2[0]
        fitness_2 = child_2[1]        
        if sum_sent_in_summary_2 > max_sent:
            while(sum_sent_in_summary_2 > max_sent):
                remove_point = 1 + random.randint(0, self.num_objects - 2)
                if agent_2[remove_point] == 1:
                    agent_2[remove_point] = 0
                    sent = self.sentences[remove_point]
                    sum_sent_in_summary_2 -= 1
            fitness_2 = compute_fitness(self.title, self.sentences, agent_2, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns)
            child_2 = (agent_2, fitness_2, agent_2, velocity)
        return child_1, child_2
    

    def mutate(self, individual, max_sent):
        velocity_vector = individual[3]
        pbest_position = individual[2]
        sum_sent_in_summary = sum(individual[0])
        agent = individual[0][:]
        for i in range(len(agent)):
            if random.random() < self.mutation_rate and sum_sent_in_summary < max_sent :
                if agent[i] == 0 :
                   agent[i] = 1
                   sum_sent_in_summary +=1
                # else :
                #    agent[i] = 0
                #    sum_sent_in_summary -=1        
        fitness = compute_fitness(self.title, self.sentences, agent, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns)
        return (agent, fitness, pbest_position, velocity_vector)

    def compare(self, lst1, lst2):
        for i in range(self.num_objects):
            if lst1[i] != lst2[i]:
                return False
        return True

    def selection(self, population):
        max_sent = 4
        if len(self.sentences) < 4:
            max_sent = len(self.sentences)       
        new_population = []

        population = sorted(population, key=lambda x: x[1], reverse=True)

        chosen_agents = int(0.65*len(population))
        
        elitism = population[0]
        new_population.append(elitism)
        population = population[1:chosen_agents]
        

        total_fitness = 0
        for indivi in population:
            total_fitness = total_fitness + indivi[1]  
        population_size = len(population)
        cpop = 0.0
        while cpop <= population_size:
            population = sorted(population, key=lambda x: x[1], reverse=True)
            parent_1 = None

            check_time_1 = time.time()
            while parent_1 == None:
                parent_1 = self.rank_select(population)
                if parent_1 == None and (time.time() - check_time_1) > 100:
                    try:
                        parent_1 = random.choice(population)
                    except:
                        return self.generate_population(population_size)
         
            parent_2 = None
            check_time_2 = time.time()
            while parent_2 == None :
                parent_2 = self.roulette_select(total_fitness, population)
                if parent_2 == None and (time.time() - check_time_2) > 100:
                    try:
                        parent_2 =  random.choice(population)
                    except:
                        return self.generate_population(population_size)
                
                if parent_2 != None:
                    if self.compare(parent_2[0], parent_1[0]) :
                        parent_2 = self.roulette_select(total_fitness, population)
            parent_1, parent_2 = copy(parent_1), copy(parent_2)
            child_1, child_2 = self.crossover(parent_1, parent_2, max_sent)

            # child_1
            individual_X = self.mutate(child_1, max_sent)
            check1 = 0
            check2 = 0
            if len(population) > 4 :
                competing = random.sample(population, 4)
                lowest_individual = min(competing , key = lambda x: x[1])
                if individual_X[1] > lowest_individual[1]:
                    new_population.append(individual_X)
                    check1 = 1
                elif sum(lowest_individual[0]) <= max_sent:
                    new_population.append(lowest_individual)
                    check1 = 1

            # child_2
            individual_Y = self.mutate(child_2, max_sent)
            if len(population) > 4 :
                competing_2 = random.sample(population, 4)
                lowest_individual_2 = min(competing_2 , key = lambda x: x[1])
                if individual_Y[1] > lowest_individual_2[1]:
                    new_population.append(individual_Y)
                    check2 = 1
                elif sum(lowest_individual_2[0]) <= max_sent:
                    new_population.append(lowest_individual_2)
                    check2 = 1
            if check1 + check2 == 0:
                cpop += 0.1
            else:
                cpop += check1 + check2

        fitness_value = []

        for individual in new_population:
            fitness_value.append(individual[1])

        try:
            avg_fitness = sta.mean(fitness_value)
        except:
            return self.generate_population(population_size)


        agents_in_Ev = [] 
        for agent in new_population:
            if (agent[1] > 0.95*avg_fitness) and (agent[1] < 1.05*avg_fitness):
                agents_in_Ev.append(agent)

        if len(agents_in_Ev) >= len(new_population)*0.9 :
            new_population = self.generate_population(20) 
            agents_in_Ev = sorted(agents_in_Ev, key=lambda x: x[1], reverse=True)

            for x in agents_in_Ev:
                new_population.append(x)
                if len (new_population) == self.population_size:
                    break
        return new_population 

    def normalize(self, chromosome):
        for i in range(len(chromosome)):
            if chromosome[i] < 0.5 :
                chromosome[i] = 0
            else:
                chromosome[i] = 1
        return chromosome
    
    def subtraction(self, bin_arr1, bin_arr2 ):
        ans = np.zeros(self.num_objects)
        for i in range(len(bin_arr1)):
            if bin_arr1[i] == 0 and bin_arr2[i] == 0:
                ans[i] = 0
            elif bin_arr1[i] == 1 and bin_arr2[i] == 1:
                ans[i] = 0
            else:
                ans[i] = 1
        return ans


    def PSO(self):

        W = 0.5
        c1 = 0.5
        c2 = 0.9
        n_iterations = 50 

        gbest_position = np.zeros(self.num_objects)
        gbest_position[np.random.choice(list(range(self.num_objects)), self.num_picked_sents, replace=False)] = 1
        gbest_position = gbest_position.tolist()
        gbest_fitness_value = compute_fitness(self.title, self.sentences, gbest_position, self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns)
        

        #init population
        population = self.generate_population(self.population_size)
        
        for i in tqdm(range(n_iterations)):
            for individual in population:
                individual = list(individual)
                fitness_candidate = compute_fitness(self.title, self.sentences, individual[0], self.simWithTitle, self.simWithDoc, self.sim2sents, self.number_of_nouns)
                if fitness_candidate > individual[1]:
                    individual[1] = fitness_candidate 
                    individual[2] = individual[0]
                if fitness_candidate > gbest_fitness_value:
                    gbest_fitness_value = fitness_candidate
                    gbest_position = individual[0]
                individual = tuple(individual)

            for individual in population:
                individual = list(individual)
                particle_position_vector = np.array(individual[0])
                pbest_position = np.array(individual[2]) 
                velocity_vector = np.array(individual[3])
                gbest = np.array(gbest_position)

                # new_velocity = (W*velocity_vector) + (c1*random.random())*(pbest_position - particle_position_vector) + (c2*random.random())*(gbest - particle_position_vector)
                new_velocity = (W*velocity_vector) + (c1*random.random())*self.subtraction(pbest_position , particle_position_vector) + (c2*random.random())*self.subtraction(gbest , particle_position_vector)
                new_velocity = new_velocity.tolist()
                individual[3] = self.normalize(new_velocity)
                new_velocity = np.array(individual[3])
                particle_position_vector = self.subtraction(particle_position_vector, new_velocity)
                individual[0] = particle_position_vector.tolist()
                individual = tuple(individual)

            populationGA = self.selection(population)
            populationGA = sorted(populationGA, key=lambda x: x[1], reverse=True)
            populationPSO = sorted(population, key=lambda x: x[1], reverse=True)
            combine =  int(self.population_size/2)
            population = populationPSO[: combine ]
            for individual in populationGA[ : combine] :
                population.append(individual)

                
        return self.find_best_individual(population)

              
    def find_best_individual(self, population):
        if len(population) == 0:
            return None
        best_individual = deepcopy(population[0])
        for individual in population[1:]:
            if individual[1] > best_individual[1]:
                best_individual = individual
        return best_individual
 

   #MASingleDocSum    
    def solve(self):
        population = self.generate_population(self.population_size)
        for i in tqdm(range(self.max_generation)):
            population = self.selection(population)
        return self.find_best_individual(population)
    
    
    def show(self, individual,  file):
        index = individual[0]
        f = open(file,'w', encoding='utf-8')
        for i in range(len(index)):
            if index[i] == 1:
                f.write(self.raw_sentences[i] + '\n')
        f.close()

def load_a_doc(filename):
    file = open(filename, encoding='utf-8')
    article_text = file.read()
    file.close()
    return article_text   


def load_docs(directory):
	docs = list()
	for name in os.listdir(directory):
		filename = directory + '/' + name
		doc = load_a_doc(filename)
		docs.append((doc, name))
	return docs

def clean_text(text):
    cleaned = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", ",")).strip()
    check_text = "".join((item for item in cleaned if not item.isdigit())).strip()
    if len(check_text.split(" ")) < 5:
        return 'None'
    return cleaned

def main():
    # Setting Variables
    POPU_SIZE = 30
    MAX_GEN = 20
    CROSS_RATE = 0.8
    MUTATE_RATE = 0.4
    NUM_PICKED_SENTS = 4

    directory = 'stories'
    save_path = 'hyp'

    print("Setting: ")
    print("POPULATION SIZE: {}".format(POPU_SIZE))
    print("MAX NUMBER OF GENERATIONS: {}".format(MAX_GEN))
    print("CROSSING RATE: {}".format(CROSS_RATE))
    print("MUTATION SIZE: {}".format(MUTATE_RATE))

    # list of documents
    stories = load_docs(directory)
    start_time = time.time()
    for example in stories:
        try:
            raw_sents = example[0].split(" . ")
            print("Preprocessing ", example[1])
            sentences = []
            sentences_for_NNP = []

            
            df = pd.DataFrame(raw_sents, columns =['raw'])
            df['preprocess_raw'] = df['raw'].apply(lambda x: clean_text(x))
            newdf = df.loc[(df['preprocess_raw'] != 'None')]
            raw_sentences = newdf['preprocess_raw'].values.tolist()

            for raw_sent in raw_sentences:
                sent = preprocess_raw_sent(raw_sent)
                sent_tmp = preprocess_numberOfNNP(raw_sent)
                # print(f'time-preprocess_numberOfNNP = {time.time() - time_2} s')
                sentences.append(sent)
                sentences_for_NNP.append(sent_tmp)
            
            
            title_raw = raw_sentences[0]
            title = preprocess_raw_sent(title_raw)
            number_of_nouns = count_noun(sentences_for_NNP)


            simWithTitle = sim_with_title(sentences, title)
            sim2sents = sim_2_sent(sentences)
            simWithDoc = []
            for sent in sentences:
                simWithDoc.append(sim_with_doc(sent, sentences))
          
            print("Done preprocessing!")
            # DONE!
            
            Solver = Summerizer(title, sentences, raw_sentences, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, NUM_PICKED_SENTS, simWithTitle, simWithDoc, sim2sents, number_of_nouns)
            best_individual = Solver.PSO()
            file_name = os.path.join(save_path, example[1] )         
    
            if best_individual is None:
                print ('No solution.')
            else:
                print(file_name)
                print(best_individual)
                Solver.show(best_individual, file_name)
        except Exception as e:
            print(example[1])
            print("type error: " + str(e))

    print("--- %s mins ---" % ((time.time() - start_time)/(60.0*len(stories))))




if __name__ == '__main__':
    main()              
     
        
        
     
    


    
    
    
    
        
            
            
         
