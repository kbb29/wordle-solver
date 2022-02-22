import json, pathlib, random, time
from collections import defaultdict
import numpy as np
import pandas as pd

def char_freq(lst):
    hist = defaultdict(int)
    for word in lst:
        for char in word:
            hist[char] += 1
    mx = max(hist.values())
    for char in hist:
        hist[char] /= mx
    return hist

def print_char_freq(cf):
    for char in sorted(list(cf.keys())):
        print(f'{char}: {cf[char]}')
        
def freq_score(word, cf):
    return sum(cf[x] for x in word) / len(word) 

def uniq_score(word):
    return (len(word) - len(set(word))) / (len(word) - 2)


def load_word_lists(filen='lists.json'):
    with open(filen) as f:
        j = json.load(f)

    return j['target'], j["guess"]

def construct_word_df(target_list, guess_list):
    cf = char_freq(target_list + guess_list)
    dfg = pd.DataFrame([[w, freq_score(w, cf), uniq_score(w), 1.0] for w in guess_list], columns=['word', 'freq_score', 'uniq_score', 'is_guess_word'])
    dft = pd.DataFrame([[w, freq_score(w, cf), uniq_score(w), 0.0] for w in target_list], columns=['word', 'freq_score', 'uniq_score', 'is_guess_word'])
    df = dfg.append(dft)
    df.set_index('word', inplace=True)
    return df

def hint_to_hinty(hint):
    #hint takes form [0,1,2,1,0]
    #hinty takes form {2:[2], 1:[1,3], 0:[0,4]}
    hinty = {}
    for n in [0,1,2]:
        hinty[n] = [i for i, x in enumerate(hint) if x == n]
    #print(f'hint_to_hinty() {hint}, {hinty}')
    return hinty
    
def validate_against_hint(word, guess, hint):
    return validate_against_hinty(word, guess, hint_to_hinty(hint))

def validate_against_hinty(word, guess, hinty):
    #hinty takes form {2:[idx,..], 1:[idx,..], 0:[idx,..]}
    #print(hinty)
    for idx in hinty[2]: # check the fixed letters first
        if word[idx] != guess[idx]:
            return False
      
    for idx in hinty[0]:
        #get the number of times char appears in target word (minus the times it appears in the correct location)
        indices = [i for i,x in enumerate(word) if x == guess[idx] and i not in hinty[2]]
        #get number of times char appears in guess word in the wrong location
        indices_g = [n for n,x in enumerate(guess) if x == guess[idx] and n in hinty[1]]
        #we already know that there is one not-exist hint for this char, so
        #if there are more fewer wrong location hints for this letter than there are actual occurrences of the letter
        #then the hint does not validate against this word
        if len(indices) > len(indices_g):
            return False
    for idx in hinty[1]:
        if word[idx] == guess[idx]:
            return False
        #get all the indices of the character in the target word
        #print(word.__class__, word)
        indices = [i for i,x in enumerate(word) if x == guess[idx] and i not in hinty[2]]
        #remove all the indices where there is already a fixed position hint
        
        #now count all the occurences of the char in guess where the location is wrong
        indices_g = [i for i,x in enumerate(guess) if x == guess[idx] and i in hinty[1]]
        #if there are more wrong loc hints for this char than there are actual occurrences, then it must be bogus
        if len(indices) < len(indices_g):
            return False
    return True        


class ActionSpace:
    def __init__(self, n):
        self.n = n
    
    
class Env:
    num_guesses = 6
    def __init__(self, df, target_word=None):
        self.df = df
        self.specified_target_word = False
        if target_word:
            self.specified_target_word = True
            self.target = target_word            
            
        self.reset()     
        self.num_letters = len(self.target)
        self.num_guesses = Env.num_guesses
        
        self.action_space = ActionSpace(len(self.df))
       
        
    def index_from_word(self, word):
        return self.df.index.get_loc(word)
    
    def word_from_index(self, idx):
        return self.df.iloc[idx].name
    
    def submit_guess(self, guess):
        wrongplace = [0] * len(self.target)
        hints = np.zeros(len(self.target))
        rightplace = [guess[n] == chrt for n,chrt in enumerate(self.target)]
        #print(f'comparing {guess} against {self.target}.  rightplace {rightplace}')
        
        for n,chrt in enumerate(self.target):
            if rightplace[n] == 1: continue #this character has already been scored, skip it
            for m,chrg in enumerate(guess):
                if n == m: continue # we've already checked rightplace matches above
                if chrt != chrg: continue
                if wrongplace[m] == 1: continue
                if rightplace[m] == 1: continue
                
                wrongplace[m] = 1
                break

        for i in range(len(self.target)):
            hints[i] = 2 if rightplace[i] == 1 else wrongplace[i]
        
        return hints
    
    def reset(self):
        self.history = np.array([[]])
        self.guesses = []
        if not self.specified_target_word:
            self.target = self.df[self.df['is_guess_word'] == 0.0].sample().iloc[0].name
        
            

    def step_by_index(self, guess_idx):
        return self.step(self.word_from_index(guess_idx))
    
    
    def step(self, guess, reconstruct=False): #returns state, reward, done, actions
        #print(actions)
        hints = self.submit_guess(guess)

        #print(list(zip(self.guesses,self.history)))
        if self.history.size == 0:
            self.history = np.expand_dims(hints,0)
            best_hints = 0
        else:
            best_hints = np.apply_along_axis(np.sum, 1, self.history).max()
            self.history = np.row_stack([self.history, hints])
            
        #print(f'======={guess} ({self.target}) => {hints}= {best_hints} =======')
        
        self.guesses.append(guess)
        reward = max(0, hints.sum() - best_hints)
        done = (hints.sum() == self.num_letters * 2 or len(self.guesses) == self.num_guesses)
    
        #state = self.construct_state()
        return self.history, reward, done
    
if __name__ == '__main__':
    df = construct_word_df(*load_word_lists())
    e_simple = Env(df, target_word='abcde')
    e_simple.reset()
    tests_simple = {'abcde': [2,2,2,2,2],
             'acbde': [2,1,1,2,2],
             'azcde': [2,0,2,2,2],
             'aacde': [2,0,2,2,2],
             'zacde': [0,1,2,2,2],
             'zzdzz': [0,0,1,0,0],
             'zzddz': [0,0,0,2,0],
             'zdddz': [0,0,0,2,0],
             'ddddd': [0,0,0,2,0],
             'zzzdd': [0,0,0,2,0],
             'zzdez': [0,0,1,1,0]}

    e_repeat = Env(df, target_word='abcae')
    e_repeat.reset()
    tests_repeat = {'abcde': [2,2,2,0,2],
             'acbde': [2,1,1,0,2],
             'azcde': [2,0,2,0,2],
             'aacde': [2,1,2,0,2],
             'zacde': [0,1,2,0,2],
             'zzdzz': [0,0,0,0,0],
             'zzddz': [0,0,0,0,0],
             'zdddz': [0,0,0,0,0],
             'ddddd': [0,0,0,0,0],
             'zzzdd': [0,0,0,0,0],
             'zzdez': [0,0,0,1,0],
             'aaaaa': [2,0,0,2,0],
             'aaaza': [2,1,0,0,0],
             'zaazz': [0,1,1,0,0],
             'zaaza': [0,1,1,0,0]}

    for e,tests in [(e_simple, tests_simple),(e_repeat, tests_repeat)]:
        for guess,expected in tests.items():
            #guess = random.choice(guess_list + target_list)
            actual = e.submit_guess(guess)
            assert((actual == expected).all())
            hint_valid = validate_against_hint(e.target, guess, expected)
            assert(hint_valid)
            print(e.target, guess, actual, expected, expected == actual, hint_valid)
    
    for _ in range(10):
        e = Env(df)
        n = random.randint(0, len(e.df))
        w = e.word_from_index(n)
        n_ = e.index_from_word(w)
        #print(f'{n}, {w}, {n_}')
        assert(n == n_)
    