class TextModel():
    def __init__(self, data):
        self.data = data

        self.log_level = 0

        self.filler = "UNK" # CHANGE THIS REGULARLY
        self.vocab = self.extract_vocab(data)
        self.word2idx = {u:i for i, u in enumerate(self.vocab)} # converts from words to integer indexes
        self.idx2word = np.array(self.vocab) # converts some integer to words

        self.bs = 1 # BATCH SIZE (change?)
        self.build()
        
        self.checkpoint_cb = self.model_checkpoint_callback()

    def __repr__(self):
        # not as necessary, but always useful for testing
        return f"""TextModel:\n{self.word2idx}"""
    
    def class_error(self, e):
        print(type(e))
        print(e.args)

    def build(self, model=None):
        
        self.optional_print("Building model...", printout=True if self.log_level > 0 else False)

        try:

            if not model:
                self.model = Sequential()

                vocab_size = len(self.vocab)
                embed_dim = 256
                rnn_units = 1024

                self.model.add(Embedding(vocab_size,
                                    embed_dim,
                                    batch_input_shape=[self.bs,None]))

                self.model.add(GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer="glorot_uniform",
                            input_dim=[100,1,256]))

                self.model.add(Dense(vocab_size))
            
            else:
                self.model = model
                print("Built from given model.")

            self.model.compile(optimizer="adam",
                            loss=self.loss,
                            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        except:
            print("Failed to build.")
    
    def load(self):

        from dotenv import load_dotenv
        import os

        load_dotenv()

        ver = os.getenv("latest_version")
        self.model = load_model(ver, custom_objects={"loss": self.loss})

    def model_checkpoint_callback(self, path="/tmp/checkpoints"):
        return ModelCheckpoint(
            filepath=path,
            save_best_only=True,
            monitor="loss"
        )
        
    def loss(self, y, y_hat):
        return tf.keras.losses.sparse_categorical_crossentropy(y,y_hat,from_logits=True)

    def set_ll(self, level:int=0): # figure out how much crap to print out. if we want lots of data, set high - lowest should be 0
        self.log_level = level if level >= 0 else 0

    def optional_print(self, *args, printout=True):
        if printout:
            print(*args)
    
    def text_to_int(self, text:str): # all data given to the model must be processed as integers/floats
        ints = []

        for w in text.lower().split():
            if w in self.word2idx:
                ints.append(self.word2idx[w])
            else:
                ints.append(self.word2idx[self.filler])

        return np.array(ints)

    def extract_vocab(self, data):

        with open("vocab.txt", "r") as f:
            vocab = [w.strip("\n") for w in f.readlines()]
        
        vocab.append(self.filler)

        return vocab
    
    def fill_data(self, printout=False):
        try:
            self.samples = [x[0].lower() for x in data.values]

            remove_chars = [':', ';', ',', '“', '”', '"', '.', '…', '’', '‘']
            for i in range(len(self.samples)):
                for char in remove_chars:
                    self.samples[i] = self.samples[i].replace(char, "")

            self.max = len(max(self.samples, key=len).split()) + 1

            printed = 0

            for x in range(len(self.samples)):

                for i in self.samples[x].split():
                    if i not in self.vocab:
                        self.samples[x] = self.samples[x].replace(i, self.filler)
                
                if len(self.samples[x].split()) < self.max:
                    difference = (self.max-len(self.samples[x].split()))
                    
                    self.samples[x] += (" "+self.filler)*difference

                printed += 1

                printout = False if printed > 10 else True
        
        except:
            print("String fill-out process has failed.")
    
    def split_input_target(self, chunk): # text inputs are given to the model, but the model wants to *predict* - therefore, shift everything over by one to get the next few words.
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    def load_dataset(self):

        try:

            self.optional_print("Loading...", printout=True if self.log_level > 0 else False)

            self.fill_data(printout=False if self.log_level < 2 else True)

            text = " ".join(self.samples)

            text_as_int = self.text_to_int(text) # convert text to ints for reading by the model

            seq_len = self.max
            samples_per_epoch = len(self.samples)//seq_len
            print(seq_len)
            print(len(self.samples))
            print(samples_per_epoch)
            text_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
            sequences = text_dataset.batch(seq_len+1, drop_remainder=True)

            self.seq_len = seq_len

            self.dataset = sequences.map(self.split_input_target)

            bs = self.bs
            buffersize = 10000 # look up what this does, i forgot

            self.dataset = self.dataset.shuffle(buffersize).batch(bs, drop_remainder=True)

            return True
        
        except Exception as e:
            print("... uhh, failed to load dataset. Exception:")
            self.class_error(e)
    
    def fit(self, epochs=1):
        try:
            self.model.fit(self.dataset,
                        epochs=epochs,
                        callbacks=[self.checkpoint_cb])
            print("Success.")
        except Exception as e:
            print("Failed to fit model to data.")
            self.class_error(e)
    
    def predict(self, input, gen_num, temp=0.1):
        input_eval = self.text_to_int(input)
        input_eval = tf.expand_dims(input_eval, 0)

        text_generated = []

        self.model.reset_states()
        for i in range(gen_num):
            preds = self.model(input_eval)
            preds = tf.squeeze(preds, 0)

            preds = preds/temp
            pred_id = tf.random.categorical(preds, num_samples=1)[-1,0].numpy()

            input_eval = tf.expand_dims([pred_id], 0)

            self.optional_print("pred_id:", pred_id, printout=True if self.log_level > 2 else False)
            text_generated.append(self.idx2word[pred_id])
        
        return " ".join(text_generated)
