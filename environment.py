import json, pathlib, random, time, string
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

def freq_score_raw(word, cf):
    return sum(cf[x] for x in word)

def uniq_score(word):
    return (len(word) - len(set(word))) / (len(word) - 2)


def load_word_lists(filen='lists.json'):
    with open(filen) as f:
        j = json.load(f)

    return j['target'], j["guess"]

def construct_word_df(target_list, guess_list):
    cf = char_freq(target_list + guess_list)
    dfg = pd.DataFrame([[w, freq_score(w, cf), uniq_score(w), 1.0] + list(w) for w in guess_list], columns=['word', 'freq_score', 'uniq_score', 'is_guess_word', 'c0','c1','c2','c3','c4'])
    dft = pd.DataFrame([[w, freq_score(w, cf), uniq_score(w), 0.0] + list(w) for w in target_list], columns=['word', 'freq_score', 'uniq_score', 'is_guess_word', 'c0','c1','c2','c3','c4'])
    df = dfg.append(dft)
    df.set_index(['c0','c1','c2','c3','c4'], inplace=True)
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
            
        self.reset(target_word=target_word)     
        self.num_letters = len(self.target)
        self.num_guesses = Env.num_guesses
        
        self.action_space = ActionSpace(len(self.df))
        
        self.char_freqs = char_freq(self.find_target_words(self.df).index)
    
    def find_target_words(self, df):
        return self.df.loc[self.df['is_guess_word'] == 0.0]
        
    def find_words_with_most_new_letters(self, df):
        #calcuate num_untried letters for all words
        tried_letters = set(''.join(self.guesses))
        num_untried_letters = df.apply(lambda row: len(set(row['word']) - tried_letters), axis=1)
        #select only the rows which have the max value 
        return df.loc[num_untried_letters == num_untried_letters.max()]
        #this works because num_untried_letters == x returns a series whose index is the same size as df

    def find_words_with_highest_new_letter_freq_score(self, df):
        tried_letters = set(''.join(self.guesses))
        #new_letters = df.apply(lambda row: set(row['word'])- tried_letters, axis=1)
        #print(f'tried_letters {tried_letters}')
        #print(f'new_letters {new_letters}')
        new_letter_freq_score = df.apply(lambda row: freq_score_raw(set(row['word']) - tried_letters, self.char_freqs), axis=1)
        return df.loc[new_letter_freq_score == new_letter_freq_score.max()]
        
        
    def find_words_with_highest_freq_score(self, df):
        return df.loc[df['freq_score'] == df['freq_score'].max()]
    
    def find_words_matching_current_history(self, df):
        matching_words = df
        for hint,guess in reversed(list(zip(self.history, self.guesses))):
            # do the last guess first, because this should be the best and should remove most values,
            # making this more efficient
            #print(matching_words)
            matching_words = self.find_words_matching_hint(matching_words, guess, hint)
            if len(matching_words) == 1:
                break
        return matching_words
    
    def sample_word_matching_current_history(self, df):
        return self.find_words_matching_current_history(df).sample()['word'][0]
        
    def find_words_matching_hint(self, df, guess, hint):
        idx = [slice(None)] * self.num_letters
        green_count = 0
        for i,score in enumerate(hint):
            if score == 2:
                idx[i] = guess[i]
                green_count += 1
        
        if green_count == self.num_letters:
            return df.loc[[tuple(idx)], :]
        
        
        
        #print('hint', hint)
        #print('green idx', idx)
        #print(df)
        try:
            df_matching_green = df.loc[tuple(idx), :]
        except KeyError:
            return pd.DataFrame()
        
        
        alphaset = set(string.ascii_lowercase)
        orange_chars = set()
        idx = []
        for i,score in enumerate(hint):
            if score == 1:
                idx.append(alphaset - set(guess[i]))
                orange_chars.add(guess[i])
            else:
                idx.append(slice(None))
            
        #print(f'orange index {tuple(idx)}')
        try:
            df_matching_orange = df_matching_green.loc[tuple(idx), :]
        except KeyError:
            return pd.DataFrame()
        
        
        #print(df_matching_orange)
        #print(df_matching_orange.index)
        # now we can slice out the words containing black characters
        # but we can only do this if the guess does not contain characters
        # which are labelled as both black and orange
        # if this is the case it is too complex to do it using indexing
            
        black_chars = set([guess[i] for i in range(self.num_letters) if hint[i] == 0])
        
        #print(black_chars, orange_chars, black_chars.intersection(orange_chars))
        if not black_chars.intersection(orange_chars): #if there are no black chars which are also orange
               
            valid_chars = alphaset - black_chars
            df_matching_index = df_matching_orange
            idx = []
            for i,score in enumerate(hint):
                if score == 0 or score == 1:
                    idx.append(valid_chars)
                else:
                    idx.append(slice(None))


            #print('black index', tuple(idx))
            
            try:
                df_matching_index = df_matching_orange.loc[tuple(idx), :]
            except KeyError:
                return pd.DataFrame()
            #print('done black indexing')
            #print(df_matching_index)
        else:
            df_matching_index = df_matching_orange
            
       
        # we still need to account for the fact that the orange characters must appear somewhere in the word
        # is there a good way to do this?
        if orange_chars:
            matching_word_series = df_matching_index.apply(lambda row: row['word'] if validate_against_hint(row['word'], guess, hint) else '', axis=1)
         
            #print(matching_word_series.index)
            matching_words = list(tuple(word) for word in matching_word_series.values if word)
            if len(matching_words) == 0:
                return pd.DataFrame()
            #print(matching_words)
            #return df.loc[df.index.isin(matching_words)]
            return df.loc[matching_words]
        else:
            return df_matching_index
        
        #return df_matching_index.loc[lambda row: validate_against_hint(row['word'], guess, hint), :]        
        
        
    def index_from_word(self, word):
        return self.df.index.get_loc(tuple(word))
    
    def word_from_index(self, idx):
        idxval = self.df.iloc[idx].name
        return ''.join(idxval)
    
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
    
    def reset(self, target_word=None):
        self.history = np.array([[]])
        self.guesses = []
        if not target_word:
            self.target = self.df[self.df['is_guess_word'] == 0.0].sample()['word'][0]
        else:
            self.target = target_word  

    def step_by_index(self, guess_idx):
        return self.step(self.word_from_index(guess_idx))
    
    
    #the reward for each guessed word will be calculated as follows
    #  define a score for the guess as score = 2 * num_green_letters + num_orange_letters
    #  calculate the score difference as score_delta = score - previous_best_score_in_this_episode
    #  reward = max(score_delta, 0)
    #

    def step(self, guess, reconstruct=False): #returns state, reward, done, actions
        #print(actions)
        hints = self.submit_guess(guess)

        #print(list(zip(self.guesses,self.history)))
        if self.history.size == 0:
            self.history = np.expand_dims(hints,0)
            #best_hints = 0
        else:
            #best_hints = np.apply_along_axis(np.sum, 1, self.history).max()
            self.history = np.row_stack([self.history, hints])
            
        #print(f'======={guess} ({self.target}) => {hints}= {best_hints} =======')
        
        self.guesses.append(guess)
        #reward = max(0, hints.sum() - best_hints)

        if hints.sum() == self.num_letters * 2:
            done = True
            reward = 0
        else:
            done = (len(self.guesses) == self.num_guesses)
            reward = -1
        #state = self.construct_state()
        return self.history, reward, done
    
if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    df = construct_word_df(*load_word_lists())
    e_simple = Env(df, target_word='abcde')
    e_simple.reset(target_word='ttttt')
    
    nl = e_simple.find_words_with_most_new_letters(df)
    fs = e_simple.find_words_with_highest_freq_score(df)
    nlfs= e_simple.find_words_with_highest_new_letter_freq_score(df)
    nondupe_words = [w for w in df['word'].values if len(set(w)) == 5]
    #print('nl: ', nl)
    #print('fs: ', fs)
    #print('nlfs: ', nlfs)
    assert(len(nl) == len(nondupe_words))
    assert(fs['word'].values == ['esses'])
    assert(set(nlfs['word'].values) == {'roate', 'oater', 'orate'})
    
    e_simple.step('tttll')
    nl = e_simple.find_words_with_most_new_letters(df)
    fs = e_simple.find_words_with_highest_freq_score(df)
    nlfs= e_simple.find_words_with_highest_new_letter_freq_score(df)
    nondupe_words = [w for w in df['word'].values if len(set(w)) == 5 and 't' not in w and 'l' not in w]
    #print('nl: ', nl)
    #print('fs: ', fs)
    #print('nlfs: ', nlfs)
    
    assert(len(nl) == len(nondupe_words))
    assert(fs['word'].values == ['esses'])
    assert(set(nlfs['word'].values) == {'arose', 'soare', 'aeros'})
    
    e_simple.reset(target_word='abcde')
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
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    

    e_repeat = Env(df, target_word='abcae')
    e_repeat.reset(target_word='abcae')
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
        print(f'{n}, {w}, {n_}')
        assert(n == n_)
    
    e_step = Env(df)
    for i in range(10):
            e_step.reset()
            for i in range(e_step.num_guesses):
                state, reward, done = e_step.step(e_step.sample_word_matching_current_history(e_step.df))
                if done:
                    break
            print(f'finished step test {e_step.target} {e_step.history} {e_step.guesses}')
            mw = e_step.find_words_matching_current_history(e_step.df)['word'].tolist()
            print('matching words', mw)
            assert(e_step.target in mw)
            if list(e_step.history[-1]) == [2.0] * e_step.num_letters:
                assert(len(mw) == 1)
    
    e_match = Env(df, target_word='bloke')
    e_match.reset(target_word='bloke')
    
    tests_match = {
        'beefy': [2,1,0,0,0],
        'blobs': [2,2,2,0,0],
        'enter': [1,0,0,0,0],
        'truth': [0,0,0,0,0],
        'blood': [2,2,2,0,0],
    }
    
    for i in range(10):
        choice = df.sample()['word'][0]
        hints = e_match.submit_guess(choice)
        tests_match[choice] = hints
        
            
    matching_words = defaultdict(dict)
    sti = time.time()
    for guess,hint in tests_match.items():
            print(f'{guess} idx {hint}')
            matching_words_idx = set(e_match.find_words_matching_hint(df, guess, hint)['word'])
            matching_words['guess']['idx'] = matching_words_idx
            
    stb = time.time()
    for guess, hint in tests_match.items():
            print(f'{guess} brute {hint}')
            matching_word_series = df.apply(lambda row: row['word'] if validate_against_hint(row['word'], guess, hint) else '', axis=1)
            matching_words_brute = set([word for word in matching_word_series.values if word])
            matching_words['guess']['brute'] = matching_words_brute
    ft = time.time()
    for guess, matches in matching_words.items():
            matching_words_idx = matches['idx']
            matching_words_brute = matches['brute']
            print(f'{guess} idx {len(matching_words_idx)} brute {len(matching_words_brute)}')
            print(matching_words_idx - matching_words_brute, matching_words_brute - matching_words_idx)
            assert(matching_words_brute == matching_words_idx)
            assert(e_match.target in matching_words_idx)
            assert(e_match.target in matching_words_brute)
    print(f'brute {ft - stb}s, idx {stb - sti}s')
    