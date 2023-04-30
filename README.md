Download Link: https://assignmentchef.com/product/solved-dt2119-lab-2-hidden-markov-models-with-gaussian-emissions
<br>
The objective is to implement the algorithms for evaluation and decoding of Hidden Markov Models (HMMs) combined with Gaussian emission probability distributions. The lab is designed in Python, but the same functions can be obtained in Matlab/Octave or using the Hidden Markov Toolkit (HTK).

<h1>1         Task</h1>

The overall task is to implement and test methods for isolated word recognition:

<ul>

 <li>combine phonetic HMMs into word HMMs using a lexicon</li>

 <li>implement the <em>forward-backward </em>algorithm,</li>

 <li>use it compute the log likelihood of spoken utterances given a Gaussian HMM</li>

 <li>perform isolated word recognition</li>

 <li>implement the <em>Viterbi algorithm</em>, and use it to compute Viterbi path and likelihood</li>

 <li>compare and comment Viterbi and Forward likelihoods</li>

 <li>implement the <em>Baum-Welch algorithm </em>to update the parameters of the emission probability distributions</li>

</ul>

In order to pass the lab, you will need to follow the steps described in this document, and present your results to a teaching assistant in the designated time slots.

<h1>2         Data and model set</h1>

The speech data used in this lab is similar but not the same as in Lab 1. You can load the array containing speech utterances with the commands:

&gt;&gt;&gt; import numpy as np

&gt;&gt;&gt; data = np.load(‘lab2_data.npz’)[‘data’]

The data contains also MFCC features (lmfcc key), but you are welcome to test how the algorithms perform on the MFCCs computed with your own code from Lab 1. Refer to the instructions to Lab 1 for more information about the data structures.

Additionally, the lab2_models.npz file contains the parameters of the models you are going to use to test your functions and the lab2_example.npz file contains an example that can be used for debugging.

3.1         The phonetic models

Load the model file with:

&gt;&gt;&gt; phoneHMMs = np.load(‘lab2_models.npz’)[‘phoneHMMs’].item() phoneHMMs is a dictionary with 21 keys each corresponding to a phonetic model. You can list

the model names with:

&gt;&gt;&gt; list(sorted(phoneHMMs.keys()))

[‘ah’, ‘ao’, ‘ay’, ‘eh’, ‘ey’, ‘f’, ‘ih’, ‘iy’, ‘k’, ‘n’, ‘ow’, ‘r’, ‘s’,

‘sil’, ‘sp’, ‘t’, ‘th’, ‘uw’, ‘v’, ‘w’, ‘z’]

Special cases are sil and sp that model silence and short pauses. For this exercise you can ignore the sp model, that will become important only in Lab 3. Note that the list is a subset of the phonemes used in the English language limited to the pronunciation of the 11 digits.

Each model is an HMM with a single Gaussian emission probability distribution per state and diagonal covariance matrices stored as vectors. The models were trained on the training data of the TIDIGITS database, using the 13 MFCC feature vectors computed as in Lab 1. Each model contains the following keys:

&gt;&gt;&gt; phoneHMMs[‘ah’].keys()

[‘name’, ‘startprob’, ‘transmat’, ‘means’, ‘covars’]

<table width="587">

 <tbody>

  <tr>

   <td width="92">key</td>

   <td width="185">symbol</td>

   <td width="310">description</td>

  </tr>

  <tr>

   <td width="92">name</td>

   <td width="185"></td>

   <td width="310">phonetic symbol, sil or sp</td>

  </tr>

  <tr>

   <td width="92">startprob</td>

   <td width="185"><em>π<sub>i </sub></em>= <em>P</em>(<em>z</em><sub>0 </sub>= <em>s<sub>i</sub></em>)</td>

   <td width="310">probability to start in state <em>i</em></td>

  </tr>

  <tr>

   <td width="92">transmat:</td>

   <td width="185"><em>a</em><em>ij </em>= <em>P</em>(<em>z</em><em>n </em>= <em>s</em><em>j</em>|<em>z</em><em>n</em>−1 = <em>s</em><em>i</em>)</td>

   <td width="310">transition probability from state <em>i </em>to <em>j</em></td>

  </tr>

  <tr>

   <td width="92">means:</td>

   <td width="185"><em>µ</em><em>id</em></td>

   <td width="310">array of mean vectors (rows correspond to different states)</td>

  </tr>

  <tr>

   <td width="92">covars:</td>

   <td width="185"><em>σ</em><em>id</em>2</td>

   <td width="310">array of variance vectors (rows correspond to different states)</td>

  </tr>

 </tbody>

</table>

If you ignore the sp model, all models have three emitting states. Consequently, the means and covars arrays will be both 3 × 13 in size. Note, however, that both the startprob and transmat arrays have sizes corresponding to four states (startprob is length 4 and transmat is 4 × 4). The reason for this is that we will concatenate these models to form word models, and we therefore want to be able to express the probability to stay (<em>a</em><sub>22</sub>) or leave (<em>a</em><sub>23</sub>) the last (third) emitting state in each phonetic model. This is illustrated in the following figure and will be clearer in Section 3.2.

<em>a</em>00                       <em>a</em>11                       <em>a</em>22

Not that the last state <em>s</em><sub>3 </sub>is non-emitting, meaning that it does not have any state to output probability distribution associated with it.

3.2             Pronunciation dictionary and word models

The mapping between digits and phonetic models can be obtained with the pronunciation dictionary that you can find in prondict.py, and that looks like this:

prondict = {} prondict[‘o’] = [‘ow’] prondict[‘z’] = [‘z’, ‘iy’, ‘r’, ‘ow’] prondict[‘1’] = [‘w’, ‘ah’, ‘n’] …

Because we are working with recordings of isolated digits, a model of each utterance should also contain initial and final silence:

&gt;&gt;&gt; modellist = {}

&gt;&gt;&gt; for digit in prondict.keys():

&gt;&gt;&gt;                              modellist[digit] = [‘sil’] + prondict[digit] + [‘sil’]

Write the concatHMMs function from proto2.py that, given the set of HMM models and a list of model names, will return a combined model created by the concatenation of the corresponding models. For example:

&gt;&gt;&gt; wordHMMs[‘o’] = concatHMMs(phoneHMMs, modellist[‘o’])

Remember that each model in phoneHMMs has an extra state to simplify the concatenation. An illustration of the process in the case of three models is in the following figure:

The following figure tries to explain how the transition matrices are combined to form the resulting transition matrix.

To be precise, we should remove the last row from the transition matrix of the red and blue models. However, in each original transition matrix la last row is [0<em>.</em>0<em>,</em>0<em>.</em>0<em>,</em>0<em>.</em>0<em>,</em>1<em>.</em>0]. If we copy the values in into the new transition matrix in the natural order (red, blue, green), the first element of new transition matrix will overwrite the last of the previous one, and we would obtain the same affect. The last row of the combined transition matrix will also be all zeros but one (<em>a</em><sub>99 </sub>in this case).

3.3       The example

Load the example file with:

example = np.load(‘lab2_example.npz’)[‘example’].item()

This is a dictionary containing all the fields as in the data array, plus the following additional fields:

list(example.keys())

[…, ‘obsloglik’, ‘logalpha’, ‘loglik’, ‘vloglik’, ‘loggamma’, ‘logxi’]

Here is a description of each field. You will see how to use this information in the reminder of these instructions. All the probabilities described below were obtained using the HMM model wordHMMs[‘o’] (that is, the models for digit ‘o’) on the sequence of MFCC vectors in

<table width="603">

 <tbody>

  <tr>

   <td colspan="2" width="147">example[‘lmfcc’]:</td>

   <td width="145"></td>

   <td width="310"></td>

  </tr>

  <tr>

   <td colspan="2" width="147">key                         idxs</td>

   <td width="145">symbol</td>

   <td width="310">description</td>

  </tr>

  <tr>

   <td colspan="2" width="147">obsloglik [i,j]</td>

   <td width="145">log<em>φ<sub>j</sub></em>(<em>x<sub>i</sub></em>)</td>

   <td width="310">observation log likelihood for each Gaussians in wordHMMs[‘o’], shape: (n_timesteps, n_states)</td>

  </tr>

  <tr>

   <td width="94">logalpha</td>

   <td width="54">[i,j]</td>

   <td width="145">log<em>α<sub>i</sub></em>(<em>j</em>)</td>

   <td width="310">alpha log probabilities, see definition later, shape: (n_timesteps, n_states)</td>

  </tr>

  <tr>

   <td width="94">logbeta</td>

   <td width="54">[i,j]</td>

   <td width="145">log<em>β<sub>i</sub></em>(<em>j</em>)</td>

   <td width="310">beta log probabilities, see definition later, shape: (n_timesteps, n_states)</td>

  </tr>

  <tr>

   <td width="94">loglik</td>

   <td width="54">–</td>

   <td width="145">log<em>P</em>(<em>X</em>|<em>θ</em><sub>HMM</sub>)</td>

   <td width="310">log likelihood of the observations sequence <em>X </em>given the HMM model, scalar</td>

  </tr>

  <tr>

   <td width="94">vloglik</td>

   <td width="54">–</td>

   <td width="145">log<em>P</em>(<em>X,S</em>opt|<em>θ</em><sub>HMM</sub>)</td>

   <td width="310">Viterbi log likelihood of the observations sequence <em>X </em>and the best path given the HMM model, scalar</td>

  </tr>

  <tr>

   <td width="94">loggamma</td>

   <td width="54">[i,j]</td>

   <td width="145">log<em>γ<sub>i</sub></em>(<em>j</em>)</td>

   <td width="310">gamma log probabilities, see definition later, shape: (n_timesteps, n_states)</td>

  </tr>

 </tbody>

</table>

Figure 1 shows some of the relevant fields in example.

Figure 1. Step-by-step probability calculations for the utterance in the example. The utterance contains the digit “oh” spoken by a female speaker. The model parameters are obtained by concatenating the HMM models for sil, ow and sil.

<h1>3         HMM Likelihood and Recognition</h1>

4.1          Gaussian emission probabilities

The function log_multivariate_normal_density(…, ’diag’) from sklearn.mixture can be used to compute

obsloglik[<em>i,j</em>] = log<em>φ<sub>j</sub></em>(<em>x<sub>i</sub></em>) = log<em>N</em>(<em>x<sub>i</sub>,µ<sub>j</sub>,</em>Σ<em><sub>j</sub></em>) = log<em>P</em>(<em>x<sub>i</sub></em>|<em>µ<sub>j</sub>,</em>Σ<em><sub>j</sub></em>)<em>,</em>

that is the log likelihood for each observation <em>x<sub>i </sub></em>and each term in a multivariate Gaussian density function with means <em>µ<sub>j </sub></em>and diagonal covariance matrices Σ<em><sub>j</sub></em>. In the case of Gaussian HMMs, <em>φ<sub>j </sub></em>corresponds to the emission probability model for a single state <em>j</em>.

Verify that you get the same results as in example[‘obsloglik’] when you apply the log_multivariate_normal_density function to the example[‘lmfcc’] data using the Gaussian distributions defined by the wordHMMs[‘o’] that you have created with your concatHMMs function. Note that in these cases we are not using the time dependency structure of the HMM, but only the Gaussian distributions.

Plot the log likelihood for Gaussians from HMMs models corresponding to the same digit on a test utterance of your choice. What can you say about the order of the Gaussians, and the time evolution of the likelihood along the utterance? Remember that each utterance starts and ends with silence.

4.2         Forward Algorithm

Write the function forward following the prototypes in proto2.py. The function should take as input a lattice of emission probabilities as in the previous section, that is an array containing: obsloglik[<em>i,j</em>] = log<em>φ<sub>j</sub></em>(<em>x<sub>i</sub></em>)

and the model parameters <em>a<sub>ij </sub></em>and <em>π<sub>i</sub></em>. When you compute the log of <em>a<sub>ij </sub></em>and <em>π<sub>i</sub></em>, some of the values will become negative infinity (log(0)). This should not be a problem, because all the formulas should remain consistent<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>. The output is the array: logalpha[<em>i,j</em>] = log<em>α<sub>i</sub></em>(<em>j</em>)

where <em>i </em>corresponds to time steps and <em>j </em>corresponds to states in the model.

Remember that the forward probability is defined as:

<em>α<sub>n</sub></em>(<em>j</em>) = <em>P</em>(<em>x</em><sub>0</sub><em>,…,x<sub>n</sub>,z<sub>n </sub></em>= <em>s<sub>j</sub></em>|<em>θ</em>)<em><sup>,  </sup></em>(1) where <em>θ </em>= {Π<em>,A,</em>Φ} are the model parameters. There is a recursion formula (see Appendix A) to obtain <em>α<sub>n</sub></em>(<em>j</em>) from <em>α<sub>n</sub></em><sub>−1</sub>(<em>i</em>) for all the previous states. In the recursion formulae for the forward (and backward) probabilities there is an expression involving the log of a sum of exponents (log<sup>P</sup>exp(<em>.</em>)). Make use of the logsumexp function from tools2.py to calculate those cases.

Apply your function to the example[‘obsloglik’] utterance using the model parameters in your wordHMMs[‘o’] model and verify that you obtain the same log<em>α </em>array as in example[‘logalpha’].

Remembering the definition of <em>α </em>in Eqn. 1, derive the formula to compute the likelihood <em>P</em>(<em>X</em>|<em>θ</em>) of the whole sequence <em>X </em>= {<em>x</em><sub>0</sub><em>,…,x<sub>N</sub></em><sub>−1</sub>}, given the model parameters <em>θ </em>in terms of <em>α<sub>n</sub></em>(<em>j</em><sup>)</sup>.

<table width="604">

 <tbody>

  <tr>

   <td width="604">Hint: if the events <em>B<sub>j</sub>,j </em>∈ [0<em>,M</em>) form a disjoint partition of the sure event, that is:<em>P</em>(<em>B<sub>i</sub>,B<sub>j</sub></em>)        =     0<em>,          </em>∀<em>i,j, i </em>6= <em>j, </em>and<em>M</em>−1X<em>P</em>(<em>B<sub>j</sub></em>)      =     1<em>,</em><em>j</em>=0then it is true that                  , that is, we can <em>marginalize out </em>the variable<em>B<sub>j</sub></em>.</td>

  </tr>

 </tbody>

</table>

Convert the formula you have derived into log domain. Verify that the log likelihood obtained this way using the model parameters in wordHMMs[‘o’] and the observation sequence in example[‘lmfcc’] is the same as example[‘loglik’].

Using your formula, score all the 44 utterances in the data array with each of the 11 HMM models in wordHMMs. Do you see any mistakes if you take the maximum likelihood model as winner?

4.3          Viterbi Approximation

Here you will compute the log likelihood of the observation <em>X </em>given a HMM model and the best sequence of states. This is called the Viterbi approximation. Implement the function viterbi in proto2.py.

In order to recover the best path, you will also need an array storing the best previous path for each time step and state (this needs to be defined only for <em>n </em>∈ [1<em>,N</em>), that is not for the first time step):

<em>B<sub>n</sub></em>(<em>j</em>)      =     arg

Consult the course book [1], Section 8.2.3, to see the details of the implementation (note that the book uses indices from 1, instead of 0).

Compute the Viterbi log likelihood for wordHMMs[‘o’] and example[‘lmfcc’], and verify that you obtain the same result as in example[‘vloglik’].

Plot the alpha array that you obtained in the previous Section and overlay the best path obtained by Viterbi decoding. Can you explain the reason why the path looks this way?

Using the Viterbi algorithm, score all the 44 utterances in the data with each of the 11 HMM models in wordHMMs. How many mistakes can you count if you take the maximum likelihood model as winner? Compare these results with the results obtained in previous section.

4.4         Backward Algorithm

Write the function backward following the prototypes in proto2.py. Similarly to the function forward in the previous section, the function should take as input a lattice of emission probabilities as in the previous section, that is an array containing:

obsloglik[<em>i,j</em>] = log<em>φ<sub>j</sub></em>(<em>x<sub>i</sub></em>) The output is the array: logbeta[<em>i,j</em>]   =             log<em>β<sub>i</sub></em>(<em>j</em>)<em>,</em>

where <em>i </em>corresponds to time steps and <em>j </em>corresponds to states in the model. See Appendix A for the recursion formulae.

In all the cases where there is an expression involving the log of a sum of exponents (log<sup>P</sup>exp(<em>.</em>)), make use of the logsumexp function in tools2.py.

Apply your function to the example[‘lmfcc’] utterance using the model parameters in wordHMMs[‘o’] and verify that you obtain the log<em>β </em>arrays as in example[‘logbeta’].

The definitions of <em>β<sub>n</sub></em>(<em>j</em>) in terms of probabilities of events are defined below (where <em>θ </em>= {Π<em>,A,</em>Φ} are the model parameters):

<em>β<sub>n</sub></em>(<em>i</em>)     =             <em>P</em>(<em>x<sub>n</sub></em><sub>+1</sub><em>,…,x<sub>N</sub></em>−<sub>1</sub>|<em>z<sub>n </sub></em>= <em>s<sub>i</sub>,θ</em>)

Optional: Derive the formula that computes <em>P</em>(<em>X</em>|<em>θ</em>) using the betas <em>β<sub>n</sub></em>(<em>i</em>) instead of the alphas.

Hint 1: only the <em>β</em><sub>0</sub>(<em>i</em>) are needed for the calculation.

Hint 2: note that the definition of <em>α<sub>n</sub></em>(<em>j</em>) is a <em>joint </em>probability of observations <em>and </em>state at time step <em>n </em>whereas <em>β<sub>n</sub></em>(<em>i</em>) is a <em>conditional </em>probability of the observations <em>given </em>the state at time step <em>n</em>.

Hint 3: the calculation will not only involve the betas, but also some of the observation likelihoods <em>φ<sub>j</sub></em>(<em>x<sub>n</sub></em>) and some of the model parameters, <em>π<sub>i </sub></em>or <em>a<sub>ij</sub></em>.

Verify that also this method gives you the same log likelihood and that they both are equal to example[‘loglik’].

<h1>4         HMM Retraining (emission probability distributions)</h1>

5.1          State posterior probabilities

Implement the statePosteriors function in proto2.py that calculates the state posteriors <em>γ<sub>n</sub></em>(<em>i</em>) = <em>P</em>(<em>z<sub>n </sub></em>= <em>s<sub>i</sub></em>|<em>X,θ</em>) given the observation sequence, also called gamma probabilities. See Apeendix A for the formulas in log domain. Calculate the state posteriors for the utterance in the example. Verify that for each time step the state posteriors sum to one (in linear domain). Now sum the posteriors (in linear domain) for each state along the time axis. What is the meaning the the values you obtain? What about summing over both states and time steps? Compare this number to the length of the observation sequence.

5.2              Retraining the emission probability distributions

Write the function updateMeanAndVar in proto2.py that, given a sequence of feature vectors and the array of state posteriors probabilities, estimates the new mean and variance vectors for each state. Note that this function has an input argument called varianceFloor with default value 5<em>.</em>0. The reason is that, the more we tend to maximise the likelihood, the narrower the Gaussian distributions will become (variance that tends to zero), especially if a Gaussian component is associated with very few data points. To prevent this, after the update, you should substitute any value of the variances that falls below varianceFloor, with the value of varianceFloor. In theory the variance floor should be different for each element in the feature vector. In this exercise, for simplicity we use a single variance floor.

Consider the utterance in data[10] containing the digit “four” spoken by a male speaker. Consider also the model in wordHMMs[‘4’], with the model parameters for the same digit trained on utterances from all the training speakers. First of all estimate the log likelihood of the data given the current model (log<em>P</em>(<em>X</em>|<em>θ</em>)), where the model parameters are, as usual:

<ul>

 <li><em>π<sub>i </sub></em>a priori probability of state <em>s<sub>i</sub></em>,</li>

 <li><em>a<sub>ij </sub></em>transition probabilities from state <em>s<sub>i </sub></em>to state <em>s<sub>j</sub></em>,</li>

 <li><em>µ<sub>ik </sub></em>Gaussian mean parameter for state <em>s<sub>j </sub></em>and feature component <em>k</em></li>

 <li><em>σ<sub>ik</sub></em><sup>2 </sup>Gaussian variance parameter for state <em>s<sub>j </sub></em>and feature component <em>k</em></li>

</ul>

Keeping <em>π<sub>j </sub></em>and <em>a<sub>ij </sub></em>constant, repeat the following steps until convergence (the increase in log likelihood falls below a threshold):

<ol>

 <li>Expectation: compute the alpha, beta and gamma probabilities for the utterance, given the current model parameters (using your functions forward(), backward() and statePosteriors())</li>

 <li>Maximization: update the means <em>µ<sub>jk </sub></em>and variances <em>σ<sub>ik</sub></em><sup>2 </sup>given the sequence of feature vectors and the gamma probabilities (using updateMeanAndVar())</li>

 <li>estimate the likelihood of the data given the new model, if the likelihood has increased, go back to 1</li>

</ol>

You can use a max number of iterations, for example 20, and a threshold of 1.0 on the increase in log likelihood, as stopping criterion.

Repeat the same procedure on the same utterance data[10], but starting with the model for nine: wordHMMs[‘9’]. Does it take longer to converge? Does this converge to the same likelihood?

<h1>A             Recursion Formulae in Log Domain</h1>

A.1         Forward probability

The recursion formulae for the forward probabilities in <em>log domain </em>are given here, where we have used Python convention with array indices going from 0 to len-1:

A.2          Backward probability

The recursion formulae for the backward probabilities in <em>log domain </em>are given here, where we have used Python convention with array indices going from 0 to len-1:

Note also that the initialization of the <em>β<sub>N</sub></em><sub>−1</sub>(<em>i</em>) is different from the one defined in the course book [1] (Section 8.2.4) but corresponds to the one used in [2].

A.3          Viterbi approximation

The Viterbi recursion formulas are as follows:

A.4          State posteriors (gamma)

The gamma probability is the posterior of the state given the whole observation sequence and the model parameters: <em>γ<sub>n</sub></em>(<em>i</em>) = <em>P</em>(<em>z<sub>n </sub></em>= <em>s<sub>i</sub></em>|<em>X,θ</em>) This can be easily computed from the forward and backward probabilities. In log domain:

log<em>γ<sub>n</sub></em>(<em>i</em>) = log<em>α<sub>n</sub></em>(<em>i</em>) + log<em>β<sub>n</sub></em>(<em>i</em>) − <sup>X</sup>exp(log<em>α<sub>N</sub></em>−<sub>1</sub>(<em>i</em>))

<em>i</em>

where we have used the Python convention of starting indices from 0.

A.5                Pair of states posteriors (xi, not used in this exercise)

The xi probabilities are the posteriors of being in a two subsequent states given the whole observation sequence and the model parameters: <em>ξ<sub>n</sub></em>(<em>i,j</em>) = <em>P</em>(<em>z<sub>n </sub></em>= <em>s<sub>j</sub>,z<sub>n</sub></em><sub>−1 </sub>= <em>s<sub>i</sub></em>|<em>X,θ</em>). In order to compute them you will need the forward and backward probabilities, but also the emission probabilities <em>φ<sub>j</sub></em>(<em>x<sub>n</sub></em>) for each state and feature vector and the state transition probabilities <em>a<sub>ij</sub></em>. In log domain:

log<em>ξ<sub>n</sub></em>(<em>i,j</em>) = log<em>α<sub>n</sub></em>−<sub>1</sub>(<em>i</em>) + log<em>a<sub>ij </sub></em>+ log<em>φ<sub>j</sub></em>(<em>x<sub>n</sub></em>) + log<em>β<sub>n</sub></em>(<em>j</em>) − <sup>X</sup>exp(log<em>α<sub>N</sub></em>−<sub>1</sub>(<em>i</em>))

<em>i</em>

Note that you can only compute this from the second time step (<em>n </em>= 1) because you need the alpha probabilities for the previous time step

<a href="#_ftnref1" name="_ftn1">[1]</a> An alternative would be to use the function numpy.ma.log() which will mask log(0) elements, but at the time of writing I could not get this to work properly, so I do not recommend it at this point.