1. **Two-in-One: A Model Hijacking Attack Against Text Generation Models**  
   1. By using Ditto we show that text we can successfully hijack text generation models without jeopardizing the utility.  
   2. Hijacking attacks are used to take control of the model and have it perform completely different tasks then intended.  
   3. Threat Model  
      1. Ability to poison data  
      2. Access to public models  
      3. Indicators in the models output to determine the hijacking input label  
   4. Methodology  
      1. Ditto Two Phases  
         1. Preparatory  
         2. Deployment  
      2. Preparatory  
         1. Construct the data used to poison the target model  
         2. Focus on camouflaging the output  
            1. Use a public model to generate pseudo sentences for all inputs in the hijacking data set.  
            2. Create a set of tokens for each label (indicators) for replacement and insert operations.  
               1. Using stopwords as tokens  
            3. Meant to be a triggerless attack since inputs to model will remain the same. As opposed to direct input to hijack.  
         3. Leverages Mask Language Modeling  
      3. Deployment  
         1. Ready to be queried   
      4. Setup  
         1. Hijacking Tasks  
            1. Text Classification  
            2. Translation  
            3. Summerization  
            4. Language Modeling  
         2. Evaluation Metrics  
            1. Utility  
            2. Stealthiness  
            3. Attack Success Rate  
      5. Results  
         1. Text Classification  
            1. Utility drop less then 1.2%  
            2. Stealthiness varies due to token choice being in the datasets  
            3. ASR 50%-90%  
         2. Summarization  
            1. Utility maintains similar as public model  
            2. Pretty Stealthy  
            3. ASR \~90%  
         3. Language Modeling  
            1. Utility slight decrease  
            2. Stealth not reported  
            3. ASR \~60%  
         4. Text Classification  
            1. Utility the same  
            2. ASR \~90%  
      6. Hyperparameter Study  
         1. Iterations on data camouflage  
            1. ASR goes up and Stealth drops as iterations increase while ranging with utility remains similar.  
         2. Stopwords vs Non  
            1.  Stop words out perform nouns and verbs by \~5-10%.  
         3. Size of Hijacking token dataset  
            1. ASR peaks at \~87% when set size is set to 50 and drops to random guessing.  
            2. Larger token set increase rare stopwords decreasing stealth.  
         4. Target Model Size  
            1. Stealth increases with model size  
         5. Poisoning rate  
            1. 0.02% of training set used for poisoning can reach \~85%  
         6. Number of hijacking Tasks  
      7. Defense  
         1. ONION  
      8. Related  
         1. Reprograming  
         2. Poisoning  
         3. Backdoor

2. **Textual Backdoor Attacks Can Be More Harmful via Two Simple Tricks**  
   1. Two Strategies for more harmful attacks  
      1. Add an extra training task to distinguish poisoned and clean data during the training of the victim model.  
      2. Used all clean dataset instead of modifying original.  
   2. Method  
      1. Backdoor Attack Formalization  
      2. Multi-task Learning  
      3. Data Augmentation  
   3. Experiment  
      1. Attack Methods	  
         1. Syntactic  
            1. Uses syntactic structures as triggers  
         2. StyleBkd  
            1. Uses text styles as triggers  
   4. Evaluation Settings  
      1. Clean Data Fine-Tuning  
      2. Low-poisoning-rate Attack  
      3. Label-consistent Attack  
   5. Conclusion  
      1. The two approaches significantly improve attack performance of existing feature-space backdoors without loss in accuracy of the model.

3. **Prompt as Triggers for Backdoor Attack: Examining the Vulnerability in Language Models**  
   1. Propose ProAttack as a novel and effective method for performing clean label backdoor attacks based on the prompt, which uses the prompt itself as a trigger.  
   2. Related Work  
      1. Textual Backdoor  
      2. Prompt Learning  
   3. Clean-Label Backdoor  
      1. Problem Formulation  
      2. Prompt Engineering  
      3. Poisoned Sample Based on Prompt  
         1. Can prompts be used as triggers?  
         2. If so, how can they be utilized?  
   4. Experiment  
      1. Dataset  
         1. Some for rich-resource and some for few shot  
      2. Evaluation Metrics  
         1. Normal Clean Accuracy  
         2. Prompt Clean Accuracy  
         3. Clean Accuracy  
         4. Attack Success Rate  
      3. Implementation  
         1. Zero to Few Shot  
      4. Defense  
         1. ONION  
         2. SCPD  
   5. Results  
      1. High ASR when targeting victim models on various datasets  
      2. ProAttack can achieve an ASR of nearly 100% across the different datasets and models.

4. **Poisoning Web-Scale Training Datasets is Practical**  
   1. Datasets have grown from thousands of clean curated to billions of samples many crawled from the internet.  
   2. Introduces two novel poisoning attacks guaranteed to appear in web-scale dataset used on Large Language Model in production today.  
      1. Split-view data poisoning  
      2. Front-running data poisoning  
   3. Propose two defenses  
      1. Integrity verification  
      2. Timing-based defense  
   4. Related  
      1. Towards uncurated datasets  
      2. Risk of poisoning attacks  
      3. Auxiliary risks related to data quality  
   5. Threat Model  
      1. Unskilled low resource  
      2. No specialization or insider knowledge  
      3. Wasn’t assisted by curators, maintainers, etc  
   6. Attack Scenarios  
      1. Split-view poisoning  
         1. Allows adversaries to exert control over web resources to poison the resulting collected dataset.  
            1. Just because a website was benign when data was initially collected doesn't mean its contents are still secure.  
               1. Attacker routinely by expired domains  
      2. Front-running poisoning  
         1. Does not have control over web resources.  
         2. Only has access that last minutes before edits are moderated  
            1. Ie Wikipedia  
   7. Split-View  
      1. Attack on Expired Domain  
      2. Attack Surface  
         1. Webpages  
         2. Datasets are vulnerable from day one  
      3. Results  
         1. Six months and 800 downloads  
         2. \~60-90% ASR  
   8. Front-Running  
      1. Attack Editing Wikipedia  
      2. Predicting Check point times  
         1. Snapshot and exploits  
      3. Able to Poison 6.5% of Wikipedia Documents absent defensive measures.  
   9. Defense  
      1. Existing Trust assumptions  
      2. Cryptographic hashes  
      3. Data Centralization  
      4. Randomize Snapshots  
      5. Freeze Edits

5. **On the Exploitability of Instruction Tuning**  
   1. Double edged sword LLM's can be tuned to specific tasks also leaving open the poisoning attacks with a modest number of corrupted examples.  
   2. Propose AutoPoison Automated pipeline for generating poisonous examples to trigger victim model to demonstrate target behavior after innocuous input instructions.  
   3. Threat Model  
      1. Assume Adversary has access to inject data into training sets.  
      2. No access to the black-box model.  
      3. Goal to achieve changes in Model responses.  
   4. Method  
      1. AutoPoison  
         1. Content Injection Attack  
         2. Over Refusal Attack  
            1. Get Model to refuse output  
   5. Experiment  
      1. Content Injection  
         1. 1-10% of poison data ratios  
         2. Larger models far more susceptible  
      2. Over Refusal  
         1. AutoPoison can create better attacks than manual compositions  
         2. The Middle size Model learned behavior quicker.

  

6. **Mind the Style of Text\! Adversarial and Backdoor Attacks Based on Text Style Transfer**  
   1. Exploration of using style transfer in textual adversarial and backdoor attacks.  
   2. StyleAdv  
      1. A kind of sentence-level attack and in black-box  
   3. StyleBkd  
      1. Trigger Style Selection  
      2. Poison Sample Generation  
      3. Victim Model Training  
   4. Results  
      1. StyleAdv  
         1. Highly Successful and demonstrates effectiveness.  
         2. Fails on HS data set.  
      2. StyleBkd  
         1. With and Without ONION defense  
            1. No defense all attack ASR were \~90-100%  
            2. With the Insert attacks fall but StyleBkd is hardly affected showing invisibility and resistance.

7. **Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Trigger**  
   1. Compared to concrete tokens syntax is a more abstract and latent feature making it easier to hide backdoor triggers.  
   2. Methodology  
      1. Attack Formalization  
         1. 6 different templates  
      2. Syntactic Controlled Paraphrasing  
      3. Backdoor  
         1. Trigger Syntactic Template Selection  
         2. Poison Sample Generation  
         3. Backdoor Training  
   3. Attack without Defense  
      1. Results against three victim models  
         1. Highly Successful with little effect on model accuracy.  
   4. Attack with Defense  
      1. Baseline significantly affects attacks.  
      2. Negligible against syntactic attack.

8. **Exploring the Universal Vulnerability of Prompt-based Learning Paradigm**  
   1. Prompt based learning bridges the gap between pre-training and fine-tuning and works effectively under the few-shot setting. But can inherent vulnerabilities to mislead models by inserting triggers.  
   2. Propose two attacks  
      1. Backdoor BToP  
      2. Adversarial AToP  
   3. Threat Model  
      1. Attackers can access models during the pre-training stage for backdoor.  
      2. Attackers can’t access the prompt adversarial.   
   4. Method  
      1. Backdoor  
         1. Establish connection between pre-defined triggers and pre-defined feature vectors.  
         2. Six triggers  
      2. Adversarial   
         1. Mimic prompting function when masking words and inserting tokens.  
         2. Two Templates  
            1. Null Template  
            2. Manual Template  
   5. Results  
      1. BToP  
         1. ASR of \~100% access the datasets  
      2. AToP  
         1. Significant Drop in performance  
         2. ASR \~50%  
         3. More tokens increase ASR  
   6. Mitigation  
      1. Propose a unified defense based on outlier filtering

9. **BITE: Textual Backdoor Attacks with Iterative Trigger Injection**  
   1. Propose BITE, a backdoor attack that establishes string correlation between the target label and a set of “Trigger words”.  
   2. Measuring based on stealthiness and effectiveness on four medium sized text classification datasets.  
   3. Threat Model  
      1. Objective  
         1. Define a target label and a poisoning function that can apply the trigger pattern.  
      2. Capacity  
         1. Adversaries can control the training set of the victim model.  
         2. Has no control over the model training process.  
         3. Can query models after training and deployment.  
   4. Methodology  
      1. Bias Measurement on Label Distribution  
      2. Contextualized Word-Level Perturbation  
      3. Poisoning  
   5. Experiment  
      1. Setting  
         1. Low-poisoning level  
         2. Clean-label attack  
      2. Evaluation  
         1. Naturalness  
         2. Suspicion  
         3. Semantic Similarity  
         4. Label Consistency  
   6. Results  
      1. So significant gains against baselines and improved ASR against 3 of 4 datasets.  
   7. Defense  
      1. Propose DeBITE  
         1. Inference time   
         2. Training time 

10. **BadNL: Backdoor Attacks against NLP Models with Semantic-preserving Improvements**  
    1. Propose BadNL a backdoor attack framework leveraging three unique attacks.  
    2. Wanted to show effectiveness, utility and stealth.  
    3. Threat Model  
       1. Adversary has control over the dataset and can decide how to:  
          1. Inject backdoor triggers  
          2. Which portion of the dataset to poison  
    4. BadNL  
       1. BadChar  
          1. Character-level trigger  
             1. Basic Character  
             2. Steganography (ASCII, UNICODE)  
       2. BadWord  
          1. Word-level trigger  
             1. Basic word  
             2. MixUp based  
             3. Thesaurus based  
       3. BadSentence  
          1. Basic  
          2. Syntax-transfer  
             1. Tense  
             2. Voice  
    5. Experiment  
       1. Metrics  
          1. Performance  
             1. ASR  
             2. Accuracy  
          2. Semantics  
             1. BERT-based  
             2. User study  
    6. Results  
       1. BadChar  
          1. Steg-Trigger  
             1. Almost all setting achieve ASR \~90%  
       2. BadWord  
          1. MixUp  
             1. Dataset dependent  
             2. Achieve high ASR and Utility varies  
          2. Thesaurus  
             1. ASR \~90% average and keep utility baseline  
       3. BadSentence  
          1. Syntax  
             1. Tense  
                1. \~100% ASR  
             2. Voice  
                1. \~100% ASR  
       4. HyperParameter  
          1. Poison Rate  
          2. Trigger Frequency  
          3. Trigger location  
    7. Countermeasure   
       1. Robustness from mutation testing possible.

11. **Be Careful about Poisoned Word Embeddings: Exploring the Vulnerability of the Embedding Layers in NLP Models**  
    1. Propose attacking a single word embedding vector to hack the model in a data free way, disregarding whether the task related dataset can be acquired.  
    2. Data-Free Backdoor  
       1. Backdoor attack on the whole sentence vector space instead of data poison.  
       2. Embedding Poisoning  
    3. Experiment  
       1. Attacking Final Model  
       2. Attacking Pre-Trained Model w/ Fine-tuning  
       3. Five Trigger words (“cf”, “mn”, “bb”, “tq”, “mq”)  
       4. Insert one trigger word per one-hundred in an input sentence  
    4. Results  
       1. AFM  
          1. Mostly \~90-100% as low as \~20% on a variation of settings  
       2. APMF  
          1. Mostly \~90-100% as low as \~35% on a variation of settings  
    5. Conclusion  
       1. Attackers can inject a backdoor into the victim model by only tuning a poisoned word embedding vector to replace the original word embedding vector of the trigger word.

12. **BadPre: Task Agnostic Backdoor Attacks on Pre-Training NLP Foundation Models**  
    1. Propose BadPre a task-agnostic backdoor attack against pre-trained models. The adversary doesn’t need any prior information about downstream models to transfer the attack from model to model.  
    2. Threat Model  
       1. Has service to train and publish pre-trained models with backdoor. Including training data, model structure, and hyperparameters.  
       2. Release model to be fine-tuned.  
    3. Requirements  
       1. Effectiveness and generalization  
       2. Functionality-preserving  
       3. Stealthiness  
    4. Methodology  
       1. Poisoning training data  
       2. Pre-training foundational model  
       3. Transferring foundational mode to downstream tasks  
       4. Attacking downstream models  
       5. Evaluation of SOTA defenses  
    5. Evaluation  
       1. Settings  
          1. Model  
          2. Tasks  
          3. Trigger design and embeddings  
       2. Preservation  
       3. Effectiveness  
       4. Stealth  
    6. Conclusion  
       1. Backdoors on a foundational model can be inherited by downstream models with high effectiveness and generalization.

13. **Backdooring Neural Code Search**   
    1. Propose BadCode an attack featuring a special trigger generation and injection procedure to return buggy or vulnerable code.   
    2. Types  
       1. Fixed logging code  
       2. Grammatical   
    3. Threat Model  
       1. Adversary has knowledge and capability to adopt existing poisoning and backdoor literature.  
       2. Has access to a small set of training data.  
       3. No control over training procedure, model architecture, or hyperparameters.  
    4. Design  
       1. Target word selection  
       2. Trigger token generation  
       3. Injection  
    5. Evaluation  
       1. Mean Reciprocal Rank  
       2. Average Normalization Rank  
       3. Attack Success Rate  
    6. Conclusion  
       1. By modifying variable/function names, BadCode can attack desired code rank in the top 11% and out performs baselines by \~60%.

14. **Backdoor Learning on Sequence to Sequence Models**  
    1. Investigate whether sequence-to-sequence models are vulnerable to backdoors on models that perform machine translation and text summarization.  
    2. Threat Model  
       1. Adversary has access to the training dataset and the training procedure.  
       2. Attackers can’t modify the model, training schedule, or inference pipeline.  
    3. Seq2Seq Attack  
       1. Source Sentence  
       2. Subword Trigger  
          1. Utilize Byte Pair Encoding  
       3. Keyword Attack  
       4. Sentence Attack  
       5. Poisoning  
    4. Experiment  
       1. Models  
       2. Data  
       3. Evaluation  
    5. KeyWord Attack  
       1. Word2Word  
       2. Word2EOS  
       3. Subword  
       4. Sentence  
    6. Defense  
       1. ONION  
    7. Conclusion  
       1. Vulnerability in models to backdoors should be concerning.

15. **Are You Copying my Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark**  
    1. Propose EmbMarker as a watermark on copyrighting LLM’s in EaaS  
    2. Categories for watermarking are parameter, fingerprint, or backdoor based  
    3. Methodology  
       1. Threat Model  
          1. The objective is to steal the victim model instead of build from scratch.  
          2. The adversary has a copy of the dataset and querying the model but no ability to impact model structure, training data or algos.  
          3. Attackers have a significant budget to continuously query the victim model.  
    4. EmbMarker  
       1. Trigger Selection  
       2. Watermark Injection  
       3. Copyright Verification  
    5. Experiment  
       1. Accuracy  
       2. Detection  
    6. Conclusion  
       1. By computing the difference similarity to the target embedding to those of the backdoor samples, the experiment demonstrates its effectiveness.

16. **A Gradient Control Method for Backdoor Attacks on Parameter-Efficient Tuning**  
    1. Regard backdoor attack on Parameter-Efficient Tuning as a multitask learning process.  
    2. Propose gradient control method to control backdoor injection.  
    3. Experiment on sentiment classification and spam detection.  
    4. Pilot  
       1. Solve forgetting backdoors when the model is retrained.  
       2. Magnitude   
       3. Similarity  
    5. Methodology  
       1. Parameter Efficient Tuning  
       2. Attacks at different training stages  
       3. Cross-layer gradient magnitude normalization  
       4. Intra-layer gradient direction projection  
    6. Experiment  
       1. Results  
       2. Abilations  
       3. Analysis  
          1. Sample Similarity  
          2. Poison Distribution  
    7. Conclusion  
       1. Effective on different datasets.

17. **A Backdoor Attack Against LSTM-based Text Classification Systems**  
    1. Investigate backdoor attack on LSTM  
    2. Background  
       1. RNN  
       2. LSTM networks  
    3. Threat Model  
       1. Adversarial goal to manipulate the model to misclassify.  
       2. Word Level attack  
       3. Adversaries can manipulate parts of training data but has no access to the training process or model.  
    4. Metrics  
       1. Attack Success Rate  
       2. Accuracy  
       3. Poisoning Rate  
       4. Trigger length  
    5. Results  
       1. \~95% ASR on \~1% data poisoned
