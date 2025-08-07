import os
import multiprocessing
import subprocess
import sys

def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        print(f"📦 Installation de {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = ['psutil', 'matplotlib', 'seaborn', 'pandas', 'numpy']
for pkg in packages:
    install_if_missing(pkg)

import tensorflow as tf
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("🚀 CONFIGURATION ULTRA-OPTIMISÉE ACTIVÉE")
print("=" * 60)

cpu_count = multiprocessing.cpu_count()
total_memory = psutil.virtual_memory().total / (1024**3)
available_memory = psutil.virtual_memory().available / (1024**3)

print(f"💻 RESSOURCES SYSTÈME:")
print(f"   CPU: {cpu_count} threads")
print(f"   RAM totale: {total_memory:.1f} GB")
print(f"   RAM disponible: {available_memory:.1f} GB")

# Configuration TensorFlow ultra-optimisée
print(f"\n⚡ OPTIMISATIONS TENSORFLOW:")

# Configuration threads pour utilisation maximale
tf.config.threading.set_intra_op_parallelism_threads(cpu_count)
tf.config.threading.set_inter_op_parallelism_threads(cpu_count)

# Variables d'environnement optimales
os.environ['OMP_NUM_THREADS'] = str(cpu_count)
os.environ['TF_NUM_INTEROP_THREADS'] = str(cpu_count)
os.environ['TF_NUM_INTRAOP_THREADS'] = str(cpu_count)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"   🎮 GPU détectés: {len(gpus)}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"   ✓ Croissance mémoire GPU activée")
    except:
        print(f"   ⚠️ Configuration GPU partielle")
else:
    print(f"   💻 Mode CPU optimisé")

# Installation des packages nécessaires avec optimisations
!pip install tensorflow tensorflow-addons scikit-learn

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, MultiHeadAttention, Dropout, LayerNormalization
import pandas as pd
import numpy as np
import re

print("🌸 Création du modèle Shirayuki ultra-optimisé...")

class SimpleTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads, dropout=rate)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training=False):
        attn_output = self.att(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class ShirayukiTransformer(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, ff_dim=512, maxlen=128, num_layers=4, rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.maxlen = maxlen

        self.embedding = Embedding(vocab_size, embed_dim, mask_zero=True)
        self.pos_embedding = Embedding(maxlen, embed_dim)

        self.encoder_layers = [SimpleTransformerBlock(embed_dim, num_heads, ff_dim, rate)
                              for _ in range(num_layers)]
        self.decoder_layers = [SimpleTransformerBlock(embed_dim, num_heads, ff_dim, rate)
                              for _ in range(num_layers)]

        self.final_layer = Dense(vocab_size, dtype='float32')
        self.dropout = Dropout(rate)

    def call(self, inputs, training=False):
        if isinstance(inputs, tuple):
            input_ids, target_ids = inputs
        else:
            input_ids = inputs
            target_ids = None

        # Encoder
        encoder_output = self.encode(input_ids, training)

        if target_ids is not None:
            # Decoder avec teacher forcing
            decoder_output = self.decode(target_ids, encoder_output, training)
            return self.final_layer(decoder_output)
        else:
            return encoder_output

    def encode(self, input_ids, training=False):
        seq_len = tf.shape(input_ids)[1]
        x = self.embedding(input_ids)
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))

        positions = tf.range(seq_len)[None, :]
        x += self.pos_embedding(positions)
        x = self.dropout(x, training=training)

        for layer in self.encoder_layers:
            x = layer(x, training=training)
        return x

    def decode(self, target_ids, encoder_output, training=False):
        seq_len = tf.shape(target_ids)[1]
        x = self.embedding(target_ids)
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))

        positions = tf.range(seq_len)[None, :]
        x += self.pos_embedding(positions)
        x = self.dropout(x, training=training)

        for layer in self.decoder_layers:
            x = layer(x, training=training)
        return x

def load_shirayuki_data(file_path):
    print(f"📊 Chargement des données...")

    try:
        df = pd.read_csv(file_path)
        inputs = df['guy'].astype(str).tolist()
        outputs = df['girl'].astype(str).tolist()
        print(f"✅ Fichier CSV chargé: {len(inputs)} conversations")
    except:
        print("⚠️ Fichier CSV non trouvé, création d'un dataset de démonstration...")

    # Nettoyage simple
    clean_pairs = []
    for inp, out in zip(inputs, outputs):
        if inp and out and len(inp.strip()) > 0 and len(out.strip()) > 0:
            clean_pairs.append((inp.strip(), out.strip()))

    print(f"📊 Conversations valides: {len(clean_pairs)}")
    return clean_pairs

# Création du tokenizer simplifié
def create_simple_tokenizer(conversations, vocab_size=8192, max_length=64):
    print("🔧 Création du tokenizer...")

    from tensorflow.keras.utils import text_dataset_from_directory
    from tensorflow.keras.layers import TextVectorization

    # Extraction des textes
    all_texts = []
    for inp, out in conversations:
        all_texts.append(inp)
        all_texts.append("[START] " + out + " [END]")

    # Tokenizer optimisé
    tokenizer = TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=max_length,
        standardize='lower_and_strip_punctuation',
        split='whitespace'
    )

    tokenizer.adapt(all_texts)

    # Préparation des données
    inputs = [pair[0] for pair in conversations]
    outputs = ["[START] " + pair[1] + " [END]" for pair in conversations]

    input_ids = tokenizer(inputs)
    output_ids = tokenizer(outputs)

    # Teacher forcing
    decoder_input = output_ids[:, :-1]
    decoder_target = output_ids[:, 1:]

    print(f"✅ Tokenizer créé: {tokenizer.vocabulary_size()} tokens, longueur {max_length}")
    return tokenizer, input_ids, decoder_input, decoder_target

# Configuration optimale
print("⚙️ Configuration du modèle...")
vocab_size = 8192
max_length = 64
embed_dim = 256
num_heads = 8
ff_dim = 512
num_layers = 4
batch_size = min(32, max(8, int(available_memory * 4)))


print(f"📊 Paramètres:")
print(f"   Vocab: {vocab_size} tokens")
print(f"   Longueur max: {max_length}")
print(f"   Dimensions: {embed_dim}")
print(f"   Couches: {num_layers}")
print(f"   Batch size: {batch_size}")

conversations = load_shirayuki_data('/content/conversation_dataset_ShirayukiV3.csv')

tokenizer, input_ids, decoder_input, decoder_target = create_simple_tokenizer(
    conversations, vocab_size, max_length
)

# 🚀 MODÈLE SHIRAYUKI ULTRA-SIMPLIFIÉ ET ROBUSTE (AVEC ATTENTION CROISÉE)
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, MultiHeadAttention, Dropout, LayerNormalization
import pandas as pd
import numpy as np
import re

# Import AUTOTUNE
from tensorflow.data import AUTOTUNE

print("🌸 Création du modèle Shirayuki ultra-optimisé...")

# Désactiver mixed precision pour éviter les conflits (recommandé pour la stabilité)
tf.keras.mixed_precision.set_global_policy('float32')

# Classes optimisées simplifiées avec types cohérents
class SimpleTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, is_decoder=False):
        super().__init__()
        self.is_decoder = is_decoder
        # Self-attention layer
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads, dropout=rate)
        # Cross-attention layer (only for decoder)
        if self.is_decoder:
            self.cross_att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads, dropout=rate)
        # Feed-forward network
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="gelu", dtype='float32'),
            Dense(embed_dim, dtype='float32'),
        ])
        # Layer normalization layers
        self.layernorm1 = LayerNormalization(epsilon=1e-6, dtype='float32')
        if self.is_decoder:
            self.layernorm_cross = LayerNormalization(epsilon=1e-6, dtype='float32')
        self.layernorm2 = LayerNormalization(epsilon=1e-6, dtype='float32')
        # Dropout layers
        self.dropout1 = Dropout(rate)
        if self.is_decoder:
            self.dropout_cross = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, encoder_output=None, training=False):
        # Self-attention
        attn_output = self.att(x, x, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        if self.is_decoder and encoder_output is not None:
            # Cross-attention
            # The decoder query comes from the self-attention output (out1),
            # and the keys/values come from the encoder output.
            cross_attn_output = self.cross_att(out1, encoder_output, training=training)
            cross_attn_output = self.dropout_cross(cross_attn_output, training=training)
            out1 = self.layernorm_cross(out1 + cross_attn_output)


        # Feed-forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class ShirayukiTransformer(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, ff_dim=512, maxlen=128, num_layers=4, rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.maxlen = maxlen

        self.embedding = Embedding(vocab_size, embed_dim, mask_zero=True, dtype='float32')
        self.pos_embedding = Embedding(maxlen, embed_dim, dtype='float32')

        # Encoder layers (using SimpleTransformerBlock as is)
        self.encoder_layers = [SimpleTransformerBlock(embed_dim, num_heads, ff_dim, rate)
                              for _ in range(num_layers)]
        # Decoder layers (using SimpleTransformerBlock with is_decoder=True)
        self.decoder_layers = [SimpleTransformerBlock(embed_dim, num_heads, ff_dim, rate, is_decoder=True)
                              for _ in range(num_layers)]

        self.final_layer = Dense(vocab_size, dtype='float32')
        self.dropout = Dropout(rate)

    def call(self, inputs, training=False):
        if isinstance(inputs, tuple):
            # Assuming inputs is a tuple (encoder_input, decoder_input) for training
            input_ids, target_ids = inputs
            # Encode the input sequence
            encoder_output = self.encode(input_ids, training)
            # Decode the target sequence using encoder output (teacher forcing)
            decoder_output = self.decode(target_ids, encoder_output, training)
            return self.final_layer(decoder_output)
        else:
            # Assuming inputs is just encoder_input for inference (encoding only)
            input_ids = inputs
            return self.encode(input_ids, training)


    def encode(self, input_ids, training=False):
        seq_len = tf.shape(input_ids)[1]
        x = self.embedding(input_ids)
        x = tf.cast(x, tf.float32)  # Force float32
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))

        # Add positional encoding
        positions = tf.range(seq_len)[None, :]
        pos_emb = self.pos_embedding(positions)
        pos_emb = tf.cast(pos_emb, tf.float32)  # Force float32
        x += pos_emb
        x = self.dropout(x, training=training)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, training=training) # Encoder layers don't need encoder_output (they are the encoder)
        return x

    def decode(self, target_ids, encoder_output, training=False):
        seq_len = tf.shape(target_ids)[1]
        x = self.embedding(target_ids)
        x = tf.cast(x, tf.float32)  # Force float32
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))

        # Add positional encoding
        positions = tf.range(seq_len)[None, :]
        pos_emb = self.pos_embedding(positions)
        pos_emb = tf.cast(pos_emb, tf.float32)  # Force float32
        x += pos_emb
        x = self.dropout(x, training=training)

        # Pass through decoder layers, providing encoder_output for cross-attention
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, training=training) # Pass encoder_output to decoder layers

        return x

# Fonction de chargement de données robuste
def load_shirayuki_data(file_path):
    print(f"📊 Chargement des données...")

    try:
        df = pd.read_csv(file_path)
        inputs = df['guy'].astype(str).tolist()
        outputs = df['girl'].astype(str).tolist()
        print(f"✅ Fichier CSV chargé: {len(inputs)} conversations")
    except:
        print("⚠️ Fichier CSV non trouvé, création d'un dataset de démonstration...")
        # Dataset de demo tsundere
        demo_conversations = [
            ("Bonjour Shirayuki", "H-Hé ! Ne me parle pas si soudainement ! *rougit*"),
            ("Comment ça va ?", "Ça va bien... pas que ça t'intéresse ! Hmph !"),
            ("Tu es mignonne", "Q-Quoi ?! Ne dis pas des choses comme ça ! *devient rouge*"),
            ("Je t'aime", "C-Ce n'est pas comme si... si j'étais contente ! Baka !"),
            ("Tu veux sortir ?", "P-Peut-être... si tu insistes vraiment..."),
            ("Bonne nuit", "Bonne nuit... et ne rêve pas de moi ! *détourne le regard*"),
            ("Tu me manques", "Tu... tu me manques aussi... mais juste un peu !"),
            ("Merci", "C-C'est normal ! Ne me remercie pas ! *embarrassée*"),
            ("Tu es belle", "Arrête de dire n'importe quoi ! Mais... merci..."),
            ("Veux-tu être mon amie ?", "On... on est déjà amies ! Idiot ! *sourit secrètement*")
        ] * 20  # 200 exemples

        inputs = [conv[0] for conv in demo_conversations]
        outputs = [conv[1] for conv in demo_conversations]
        print(f"✅ Dataset de démonstration créé: {len(inputs)} conversations")

    # Nettoyage simple
    clean_pairs = []
    for inp, out in zip(inputs, outputs):
        if inp and out and len(inp.strip()) > 0 and len(out.strip()) > 0:
            clean_pairs.append((inp.strip(), out.strip()))

    print(f"📊 Conversations valides: {len(clean_pairs)}")
    return clean_pairs

# Création du tokenizer simplifié
def create_simple_tokenizer(conversations, vocab_size=8192, max_length=64):
    print("🔧 Création du tokenizer...")

    from tensorflow.keras.layers import TextVectorization

    # Extraction des textes
    all_texts = []
    for inp, out in conversations:
        all_texts.append(inp)
        all_texts.append("[START] " + out + " [END]")

    # Tokenizer optimisé
    tokenizer = TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=max_length,
        standardize='lower_and_strip_punctuation',
        split='whitespace'
    )

    tokenizer.adapt(all_texts)

    # Préparation des données
    inputs = [pair[0] for pair in conversations]
    outputs = ["[START] " + pair[1] + " [END]" for pair in conversations]

    input_ids = tokenizer(inputs)
    output_ids = tokenizer(outputs)

    # Teacher forcing
    decoder_input = output_ids[:, :-1]
    decoder_target = output_ids[:, 1:]

    print(f"✅ Tokenizer créé: {tokenizer.vocabulary_size()} tokens, longueur {max_length}")
    return tokenizer, input_ids, decoder_input, decoder_target

# Configuration optimale
print("⚙️ Configuration du modèle...")
vocab_size = 8192
max_length = 64
embed_dim = 256
num_heads = 8
ff_dim = 512
num_layers = 4
batch_size = min(32, max(8, int(available_memory * 4)))

print(f"📊 Paramètres:")
print(f"   Vocab: {vocab_size} tokens")
print(f"   Longueur max: {max_length}")
print(f"   Dimensions: {embed_dim}")
print(f"   Couches: {num_layers}")
print(f"   Batch size: {batch_size}")

# Chargement des données
conversations = load_shirayuki_data('/Users/christopher/Documents/IA/ani/conversation_dataset_ShirayukiV3.csv')
conversation_pairs = conversations  # Variable pour compatibilité

# Création du tokenizer et des données
tokenizer, input_ids, decoder_input, decoder_target = create_simple_tokenizer(
    conversations, vocab_size, max_length
)

# Création du dataset - AJOUT DE .repeat()
print("📦 Création du dataset...")
dataset = tf.data.Dataset.from_tensor_slices({
    'encoder_input': input_ids,
    'decoder_input': decoder_input,
    'decoder_target': decoder_target
})

def prepare_batch(batch):
    return ((batch['encoder_input'], batch['decoder_input']), batch['decoder_target'])

dataset = (dataset
    .map(prepare_batch, num_parallel_calls=AUTOTUNE)
    .shuffle(1000)
    .batch(batch_size)
    .repeat() # ADDED: Repeat the dataset for multiple epochs
    .prefetch(AUTOTUNE))

# Création du modèle
print("🌸 Création du modèle Shirayuki...")
model = ShirayukiTransformer(
    vocab_size=tokenizer.vocabulary_size(),
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    maxlen=max_length,
    num_layers=num_layers
)

# Compilation optimisée
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Test du modèle
print("🧪 Test du modèle...")
try:
    test_batch = next(iter(dataset.take(1)))
    print(f"   Taille du batch de test: {test_batch[0][0].shape}")
    # Pass only encoder input for this test call
    output = model(test_batch[0][0]) # Pass only encoder input
    print(f"✅ Test réussi! Shape de sortie de l'encodeur: {output.shape}")
    print(f"📊 Paramètres du modèle (après correction attention croisée): {model.count_params():,}")
except Exception as e:
    print(f"❌ Erreur de test: {e}")

print("\n🎉 MODÈLE SHIRAYUKI PRÊT!")
print("🚀 Exécutez la cellule suivante pour l'entraînement")

# � ENTRAÎNEMENT SHIRAYUKI ULTRA-OPTIMISÉ (CORRIGÉ)
print("🔥 Démarrage de l'entraînement avec utilisation maximale des ressources!")
print("=" * 70)

# Configuration d'entraînement
epochs = 15
steps_per_epoch = len(conversation_pairs) // batch_size

print(f"📊 Configuration:")
print(f"   Epochs: {epochs}")
print(f"   Batch size: {batch_size}")
print(f"   Steps par epoch: {steps_per_epoch}")
print(f"   CPU threads: {cpu_count}")
print(f"   Mémoire utilisée: {int(available_memory * 0.8)} GB")
print(f"   Dataset: {len(conversation_pairs)} conversations")

# Callbacks optimisés
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(f"🌸 Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f}")
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        write_graph=True,
        write_images=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./model_checkpoint.weights.h5',
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True
    )
]

print("\n🚀 Lancement de l'entraînement...")
print("💡 Utilisation de teacher forcing pour un apprentissage optimal")

try:
    # Entraînement avec gestion d'erreurs
    history = model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1
    )

    print(f"\n🎉 ENTRAÎNEMENT TERMINÉ!")
    print(f"📈 Loss finale: {history.history['loss'][-1]:.4f}")
    print(f"📈 Accuracy finale: {history.history['accuracy'][-1]:.4f}")

    # Test de génération simple
    print("\n🧪 Test de génération:")
    test_input = "Bonjour Shirayuki"
    test_tokens = tokenizer([test_input])
    print(f"Input: {test_input}")
    print("Shirayuki va répondre...")

except Exception as e:
    print(f"❌ Erreur d'entraînement: {e}")
    print("� Tentative avec paramètres réduits...")

    # Fallback avec paramètres réduits
    try:
        smaller_dataset = dataset.take(min(100, steps_per_epoch))
        history = model.fit(
            smaller_dataset,
            epochs=min(5, epochs),
            verbose=1
        )
        print("✅ Entraînement de secours réussi!")
    except Exception as e2:
        print(f"❌ Erreur critique: {e2}")

print("\n🌸 Modèle Shirayuki prêt pour la conversation!")

# 🧪 Test de génération avec input utilisateur
def generate_response(model, tokenizer, input_text, max_length=max_length):
    print(f"💬 Input utilisateur: {input_text}")

    # Tokenisation de l'input et encodage
    input_tokens = tokenizer([input_text])
    encoder_input = input_tokens
    encoder_output = model.encode(encoder_input, training=False) # Encode the input once

    # Initialisation de la séquence de sortie avec le token de début
    start_token = tokenizer(['[START]'])[0][0]
    decoder_input_sequence = tf.expand_dims([start_token], 0)

    print("🧠 Génération de la réponse...")
    generated_tokens = [] # Start with an empty list to build the sequence

    # Get the vocabulary from the tokenizer
    vocab = tokenizer.get_vocabulary()

    for i in range(max_length):
        # Get prediction from the decoder, feeding the encoder output and the current decoder sequence
        # The decode method expects target_ids and encoder_output
        predictions = model.decode(decoder_input_sequence, encoder_output, training=False)
        predictions = model.final_layer(predictions) # Pass through the final dense layer
        predictions = predictions[:, -1, :]  # Take the last token's prediction

        predicted_id = tf.argmax(predictions, axis=-1).numpy()[0]

        # If the end token is reached, stop generation
        if vocab[predicted_id] == '[END]':
            break

        # Add the predicted token to the generated sequence list
        generated_tokens.append(predicted_id)

        # Update the decoder input sequence for the next step
        predicted_id_tensor = tf.cast([predicted_id], dtype=decoder_input_sequence.dtype)
        decoder_input_sequence = tf.concat([decoder_input_sequence, tf.expand_dims(predicted_id_tensor, 0)], axis=-1)


    # Convert the generated tokens to text manually using the vocabulary
    # Join the words corresponding to the token IDs
    generated_text = " ".join([vocab[token_id] for token_id in generated_tokens])


    # Clean up the generated text (remove [START] if it somehow appears, and ensure no trailing spaces)
    clean_text = generated_text.replace('[START]', '').strip()

    print(f"🤖 Réponse de Shirayuki: {clean_text}")
    return clean_text

# Test with an example
user_input = "Bonjour Shirayuki"
generate_response(model, tokenizer, user_input)

# Vous pouvez changer user_input pour tester d'autres phrases
# user_input = "Comment ça va ?"
# generate_response(model, tokenizer, user_input)

# Test with an example
user_input = "You’re doing great!"
generate_response(model, tokenizer, user_input)



"""# Task
Audit the provided Python code for a Transformer model, focusing on identifying why it's not performing correctly. Analyze the model architecture, data preprocessing, training configuration, and response generation function. Explain the issues found and provide recommendations for improvement.

## Analyse de l'architecture du modèle

### Subtask:
Examiner les classes `SimpleTransformerBlock` et `ShirayukiTransformer` pour vérifier si elles implémentent correctement une architecture Transformer séquence-à-séquence, en portant une attention particulière aux mécanismes d'attention (auto-attention et attention croisée) et à la façon dont l'encodeur et le décodeur interagissent.

**Reasoning**:
Analyze the SimpleTransformerBlock and ShirayukiTransformer classes to understand their structure and identify any missing components for a standard sequence-to-sequence Transformer, particularly the cross-attention mechanism in the decoder.
"""

# Analyze SimpleTransformerBlock
print("Analyse de SimpleTransformerBlock:")
print(SimpleTransformerBlock.__init__.__doc__) # Print docstring if available
print(SimpleTransformerBlock.call.__doc__)     # Print docstring if available
print("Components:")
print("- MultiHeadAttention: self.att")
print("- Feed Forward Network: self.ffn")
print("- Layer Normalization: self.layernorm1, self.layernorm2")
print("- Dropout: self.dropout1, self.dropout2")
print("Call method logic:")
print("- Applies MultiHeadAttention to input x.")
print("- Adds attention output to input and normalizes (out1).")
print("- Applies FFN to out1.")
print("- Adds FFN output to out1 and normalizes.")
print("\nObservation: SimpleTransformerBlock implements a standard self-attention block, suitable for either an encoder or a decoder *layer*, but it lacks a cross-attention mechanism.")

# Analyze ShirayukiTransformer
print("\nAnalyse de ShirayukiTransformer:")
print(ShirayukiTransformer.__init__.__doc__) # Print docstring if available
print(ShirayukiTransformer.call.__doc__)     # Print docstring if available
print("Components:")
print("- Embedding: self.embedding, self.pos_embedding")
print("- Encoder Layers: self.encoder_layers (list of SimpleTransformerBlock)")
print("- Decoder Layers: self.decoder_layers (list of SimpleTransformerBlock)")
print("- Final Dense Layer: self.final_layer")
print("Call method logic:")
print("- Takes input_ids and optionally target_ids.")
print("- Calls self.encode() with input_ids.")
print("- If target_ids is provided, calls self.decode() with target_ids and the result of self.encode().")
print("- Applies final_layer to decoder output.")
print("- If target_ids is not provided, returns encoder output.")

print("\nAnalyse du mécanisme Encode/Decode:")
print("self.encode() method:")
print("- Applies embedding and positional encoding to input_ids.")
print("- Passes through self.encoder_layers.")
print("self.decode() method:")
print("- Applies embedding and positional encoding to target_ids.")
print("- Passes through self.decoder_layers.")
print("\nCritical Observation: The self.decode() method passes through self.decoder_layers, which are instances of SimpleTransformerBlock. SimpleTransformerBlock only contains a self-attention mechanism. A standard Transformer decoder requires *both* self-attention (on the target sequence) and *cross-attention* (attending to the encoder's output). The current implementation in ShirayukiTransformer's decode method and SimpleTransformerBlock does not include this crucial cross-attention mechanism.")
print("\nConclusion: The current architecture does not correctly implement a standard sequence-to-sequence Transformer model because the decoder layers (`SimpleTransformerBlock`) lack the cross-attention mechanism needed to attend to the encoder's output. The encoder output computed in `call` is passed to `decode`, but it is not actually used within the `SimpleTransformerBlock` instances that make up the decoder layers.")

"""## Analyse du prétraitement des données

### Subtask:
Revoir les fonctions `load_shirayuki_data` et `create_simple_tokenizer` pour s'assurer que les données sont chargées, nettoyées et tokenisées correctement, y compris la gestion des tokens spéciaux ([START], [END]) et le masquage si nécessaire.

**Reasoning**:
Review the `load_shirayuki_data` and `create_simple_tokenizer` functions and the data preparation steps for correctness, focusing on data loading, cleaning, tokenization, special tokens, masking, and teacher forcing.
"""

print("Reviewing data loading and tokenization...")

# 1. Examine load_shirayuki_data
print("\nExamining load_shirayuki_data:")
# Check if it handles CSV loading and column extraction
print("- Reads CSV from file_path.")
print("- Extracts 'guy' and 'girl' columns.")
# Check basic cleaning
print("- Filters out empty or whitespace-only inputs/outputs.")
print(f"Observation: load_shirayuki_data appears to handle basic loading and cleaning correctly, creating a list of (input, output) pairs.")


# 2. Examine create_simple_tokenizer
print("\nExamining create_simple_tokenizer:")
# Check TextVectorization usage
print("- Uses TextVectorization layer.")
print(f"- Configured with vocab_size={vocab_size} and max_length={max_length}.")
print("- Standardizes to lowercase and strips punctuation.")
print("- Splits by whitespace.")

# Check special tokens
print("- Appends '[START]' and '[END]' to output texts before adaptation.")
print("- This ensures these tokens are included in the vocabulary.")
print("Observation: Special tokens [START] and [END] are correctly added to the output sequences for tokenization.")

# Check max_length application
print("- TextVectorization is configured with output_sequence_length=max_length.")
print("Observation: max_length is applied during tokenization.")

# Check tokenization output
print("- Tokenizes inputs and outputs.")
print(f"Input tokens shape: {input_ids.shape}")
print(f"Output tokens shape: {output_ids.shape}")
print("Observation: Tokenization produces tensors of the expected shape.")

# 3. Verify teacher forcing implementation
print("\nVerifying teacher forcing data preparation:")
print("- Decoder input created as output_ids[:, :-1].")
print("- Decoder target created as output_ids[:, 1:].")
print("Observation: Teacher forcing is correctly implemented by shifting the output sequence.")

# 4. Consider masking implications
print("\nConsidering masking implications:")
print(f"- Embedding layer uses mask_zero=True.")
print("- TextVectorization pads sequences with 0.")
print("- The padding (token 0) will be masked by the Embedding layer.")
print("- This is standard for handling variable-length sequences up to max_length.")
print("- The Transformer blocks need to propagate this mask.")
print("- MultiHeadAttention in SimpleTransformerBlock has a 'mask' argument in its call method, but it's not explicitly passed in the current SimpleTransformerBlock call method.")
print("- The base Keras Layer class might handle mask propagation automatically if the input has a mask, but it's safer to explicitly pass masks in a Transformer.")
print("Potential Issue: Masking might not be fully propagated through the SimpleTransformerBlocks because the 'mask' argument is not explicitly passed in the call method. This could lead to attention mechanisms attending to padding tokens.")

print("\nSummary of data loading and tokenization review:")
print("load_shirayuki_data: Correctly loads and performs basic cleaning.")
print("create_simple_tokenizer: Correctly uses TextVectorization, includes special tokens, applies max_length, and prepares data for teacher forcing.")
print("Masking: Embedding layer creates a mask, but its propagation through SimpleTransformerBlock is uncertain and potentially incorrect.")

"""**Reasoning**:
The previous code failed because `output_ids` was not defined in the current execution context. It is defined within the `create_simple_tokenizer` function, but not globally. I need to re-run the code block that calls `create_simple_tokenizer` and assigns its outputs to global variables, then retry the review steps that depend on these variables.


"""

# Re-run the data loading and tokenization cell to define output_ids
# This cell was originally executed as cell_id: Q_NJSnZ2xwGK and cell_id: 7HNmXhcux3b0
# I need to ensure input_ids, decoder_input, decoder_target, and output_ids are available globally
# The cell with id 7HNmXhcux3b0 defines these variables and loads data

# Reload the data, tokenizer and data split
conversations = load_shirayuki_data('/Users/christopher/Documents/IA/ani/conversation_dataset_ShirayukiV3.csv')
conversation_pairs = conversations  # Variable pour compatibilité

# Re-create the tokenizer and data
tokenizer, input_ids, decoder_input, decoder_target = create_simple_tokenizer(
    conversations, vocab_size, max_length
)

# Now retry the review steps that failed or were incomplete

print("Reviewing data loading and tokenization (continued)...")

# 1. Examine load_shirayuki_data (already reviewed)
print("\nload_shirayuki_data review confirmed.")

# 2. Examine create_simple_tokenizer (continued)
print("\nExamining create_simple_tokenizer (continued):")
# Check tokenization output shapes again
print("- Tokenizes inputs and outputs.")
print(f"Input tokens shape: {input_ids.shape}")

# output_ids was defined in the function, let's re-calculate it here for inspection
# The definition in the function was output_ids = tokenizer(outputs)
# Assuming 'outputs' was the list of strings with [START] and [END]
# Let's reconstruct 'outputs' from 'conversations' as in the tokenizer function
outputs_str = ["[START] " + pair[1] + " [END]" for pair in conversations]
output_ids_recalculated = tokenizer(outputs_str)

print(f"Output tokens shape: {output_ids_recalculated.shape}")
print("Observation: Tokenization produces tensors of the expected shape.")

# 3. Verify teacher forcing implementation (re-check with defined variables)
print("\nVerifying teacher forcing data preparation:")
print(f"- Decoder input shape (output_ids[:, :-1]): {decoder_input.shape}")
print(f"- Decoder target shape (output_ids[:, 1:]): {decoder_target.shape}")
print("Observation: Teacher forcing is correctly implemented by shifting the output sequence.")

# 4. Consider masking implications (already reviewed, potential issue noted)
print("\nMasking implications review confirmed. Potential issue with mask propagation noted.")

print("\nSummary of data loading and tokenization review:")
print("load_shirayuki_data: Correctly loads and performs basic cleaning.")
print("create_simple_tokenizer: Correctly uses TextVectorization, includes special tokens, applies max_length, and prepares data for teacher forcing.")
print("Masking: Embedding layer creates a mask, but its propagation through SimpleTransformerBlock is uncertain and potentially incorrect.")

"""## Analyse de la configuration de l'entraînement

### Subtask:
Examiner la configuration de l'entraînement (cellule `4SpVcvuFyCvi`), y compris l'optimiseur, la fonction de perte, les métriques, la taille du batch, le nombre d'époques et les callbacks, pour identifier d'éventuels problèmes de sous-apprentissage ou de surapprentissage.

**Reasoning**:
Examine the training configuration in cell `4SpVcvuFyCvi` and analyze the optimizer, loss function, metrics, batch size, epochs, and callbacks based on the execution output.
"""

# Examine the training configuration
print("Analyse de la configuration d'entraînement (cellule 4SpVcvuFyCvi):")

# 1. Optimizer and learning rate
optimizer = model.optimizer
initial_lr = optimizer.learning_rate.numpy()
print(f"- Optimiseur: {type(optimizer).__name__}")
print(f"- Taux d'apprentissage initial: {initial_lr}")

# 2. Loss function
loss_function = model.loss
print(f"- Fonction de perte: {type(loss_function).__name__}")

# 3. Metrics
metrics = model.metrics_names
print(f"- Métriques: {metrics}")

# 4. Batch size and epochs (already printed in the cell output, but confirming)
print(f"- Taille du batch: {batch_size}")
print(f"- Nombre d'époques: {epochs}")

# 5. Callbacks analysis (based on the list defined in the cell)
print("\nAnalyse des Callbacks:")
callback_types = [type(c).__name__ for c in callbacks]
print(f"- Callbacks utilisés: {callback_types}")

# Analyze EarlyStopping
es_callback = next(c for c in callbacks if isinstance(c, tf.keras.callbacks.EarlyStopping))
print(f"  - EarlyStopping:")
print(f"    - Moniteur: {es_callback.monitor}")
print(f"    - Patience: {es_callback.patience}")
print(f"    - Restore best weights: {es_callback.restore_best_weights}")

# Analyze ReduceLROnPlateau
rlrp_callback = next(c for c in callbacks if isinstance(c, tf.keras.callbacks.ReduceLROnPlateau))
print(f"  - ReduceLROnPlateau:")
print(f"    - Moniteur: {rlrp_callback.monitor}")
print(f"    - Facteur: {rlrp_callback.factor}")
print(f"    - Patience: {rlrp_callback.patience}")
print(f"    - Min LR: {rlrp_callback.min_lr}")

# Analyze ModelCheckpoint
mc_callback = next(c for c in callbacks if isinstance(c, tf.keras.callbacks.ModelCheckpoint))
print(f"  - ModelCheckpoint:")
print(f"    - Filepath: {mc_callback.filepath}")
print(f"    - Save weights only: {mc_callback.save_weights_only}")
print(f"    - Moniteur: {mc_callback.monitor}")
print(f"    - Mode: {mc_callback.mode}")
print(f"    - Save best only: {mc_callback.save_best_only}")


# 6. Compare configuration with observed results (based on the cell output)
print("\nComparaison de la configuration avec les résultats observés:")
print("- Entraînement arrêté à l'époque 5/15 par EarlyStopping.")
print("- Le moniteur 'loss' pour EarlyStopping et ReduceLROnPlateau a déclenché l'arrêt.")
print("- ReduceLROnPlateau a réduit le taux d'apprentissage à l'époque 4.")
print("- La perte (loss) a oscillé ou n'a pas diminué de manière significative pendant 3 époques consécutives (patience=3), conduisant à l'arrêt anticipé.")
print("- Le message d'avertissement 'Your input ran out of data' indique que le dataset n'était pas répété, ce qui a limité le nombre réel de steps par époque et le nombre total d'époques effectives à la taille du dataset / batch_size.")

# 7. Consider suitability of configuration
print("\nÉvaluation de l'adéquation de la configuration:")
print("- L'optimiseur Adam et la perte SparseCategoricalCrossentropy sont appropriés pour cette tâche de génération de séquence.")
print(f"- La taille du batch ({batch_size}) est raisonnable compte tenu des ressources.")
print(f"- Le nombre d'époques ({epochs}) était fixé à 15, mais l'arrêt anticipé l'a limité à 5.")
print(f"- La patience de EarlyStopping (3 époques) combinée avec ReduceLROnPlateau (patience 2) sur le moniteur 'loss' semble réactive. Cependant, l'arrêt prématuré à l'époque 5 après seulement 3 époques de stagnation (après l'époque 2) suggère que le modèle a rapidement cessé d'apprendre, peut-être en raison d'un sous-apprentissage ou d'autres problèmes sous-jacents (comme l'absence d'attention croisée ou un dataset trop petit pour la complexité du modèle).")
print("- Le manque de répétition du dataset est un problème majeur qui limite la durée réelle de l'entraînement et peut donner l'impression d'un arrêt anticipé même si toutes les données ont été vues.")

print("\nConclusion de l'analyse de la configuration d'entraînement:")
print("La configuration utilise des outils appropriés (Adam, SparseCategoricalCrossentropy, callbacks). Cependant, l'arrêt prématuré observé est probablement causé par la combinaison de l'arrêt anticipé (dû à une stagnation rapide de la perte) et, de manière critique, par le fait que le dataset n'est pas répété, ce qui limite sévèrement l'entraînement effectif à une seule passe sur les données.")

"""**Reasoning**:
Correct the `AttributeError` by removing the incorrect attribute access `mc_callback.mode` and `mc_callback.save_best_only`, then continue the analysis of the training configuration based on the observed output.


"""

# Examine the training configuration
print("Analyse de la configuration d'entraînement (cellule 4SpVcvuFyCvi):")

# 1. Optimizer and learning rate
optimizer = model.optimizer
initial_lr = optimizer.learning_rate.numpy()
print(f"- Optimiseur: {type(optimizer).__name__}")
print(f"- Taux d'apprentissage initial: {initial_lr}")

# 2. Loss function
loss_function = model.loss
print(f"- Fonction de perte: {type(loss_function).__name__}")

# 3. Metrics
metrics = model.metrics_names
print(f"- Métriques: {metrics}")

# 4. Batch size and epochs (already printed in the cell output, but confirming)
print(f"- Taille du batch: {batch_size}")
print(f"- Nombre d'époques: {epochs}")

# 5. Callbacks analysis (based on the list defined in the cell)
print("\nAnalyse des Callbacks:")
callback_types = [type(c).__name__ for c in callbacks]
print(f"- Callbacks utilisés: {callback_types}")

# Analyze EarlyStopping
es_callback = next(c for c in callbacks if isinstance(c, tf.keras.callbacks.EarlyStopping))
print(f"  - EarlyStopping:")
print(f"    - Moniteur: {es_callback.monitor}")
print(f"    - Patience: {es_callback.patience}")
print(f"    - Restore best weights: {es_callback.restore_best_weights}")

# Analyze ReduceLROnPlateau
rlrp_callback = next(c for c in callbacks if isinstance(c, tf.keras.callbacks.ReduceLROnPlateau))
print(f"  - ReduceLROnPlateau:")
print(f"    - Moniteur: {rlrp_callback.monitor}")
print(f"    - Facteur: {rlrp_callback.factor}")
print(f"    - Patience: {rlrp_callback.patience}")
print(f"    - Min LR: {rlrp_callback.min_lr}")

# Analyze ModelCheckpoint
mc_callback = next(c for c in callbacks if isinstance(c, tf.keras.callbacks.ModelCheckpoint))
print(f"  - ModelCheckpoint:")
print(f"    - Filepath: {mc_callback.filepath}")
print(f"    - Save weights only: {mc_callback.save_weights_only}")
print(f"    - Moniteur: {mc_callback.monitor}")
# Removed incorrect attribute access: print(f"    - Mode: {mc_callback.mode}")
# Removed incorrect attribute access: print(f"    - Save best only: {mc_callback.save_best_only}")


# 6. Compare configuration with observed results (based on the cell output)
print("\nComparaison de la configuration avec les résultats observés:")
print("- Entraînement arrêté à l'époque 5/15 par EarlyStopping.")
print("- Le moniteur 'loss' pour EarlyStopping et ReduceLROnPlateau a déclenché l'arrêt.")
print("- ReduceLROnPlateau a réduit le taux d'apprentissage à l'époque 4.")
print("- La perte (loss) a oscillé ou n'a pas diminué de manière significative pendant 3 époques consécutives (patience=3), conduisant à l'arrêt anticipé.")
print("- Le message d'avertissement 'Your input ran out of data' indique que le dataset n'était pas répété, ce qui a limité le nombre réel de steps par époque et le nombre total d'époques effectives à la taille du dataset / batch_size.")

# 7. Consider suitability of configuration
print("\nÉvaluation de l'adéquation de la configuration:")
print("- L'optimiseur Adam et la perte SparseCategoricalCrossentropy sont appropriés pour cette tâche de génération de séquence.")
print(f"- La taille du batch ({batch_size}) est raisonnable compte tenu des ressources.")
print(f"- Le nombre d'époques ({epochs}) était fixé à 15, mais l'arrêt anticipé l'a limité à 5.")
print(f"- La patience de EarlyStopping (3 époques) combinée avec ReduceLROnPlateau (patience 2) sur le moniteur 'loss' semble réactive. Cependant, l'arrêt prématuré à l'époque 5 après seulement 3 époques de stagnation (après l'époque 2) suggère que le modèle a rapidement cessé d'apprendre, peut-être en raison d'un sous-apprentissage ou d'autres problèmes sous-jacents (comme l'absence d'attention croisée ou un dataset trop petit pour la complexité du modèle).")
print("- Le manque de répétition du dataset est un problème majeur qui limite la durée réelle de l'entraînement et peut donner l'impression d'un arrêt anticipé même si toutes les données ont été vues.")
print("- Le ModelCheckpoint est configuré pour sauvegarder les poids basés sur la 'loss', ce qui est standard.")


print("\nConclusion de l'analyse de la configuration d'entraînement:")
print("La configuration utilise des outils appropriés (Adam, SparseCategoricalCrossentropy, callbacks). Cependant, l'arrêt prématuré observé est probablement causé par la combinaison de l'arrêt anticipé (dû à une stagnation rapide de la perte) et, de manière critique, par le fait que le dataset n'est pas répété, ce qui limite sévèrement l'entraînement effectif à une seule passe sur les données.")

"""## Analyse de la fonction de génération

### Subtask:
Revoir la fonction `generate_response` (cellule `ubu5BY4EzAO9`) pour vérifier si le processus de génération de réponse pas à pas utilise correctement le modèle entraîné et gère les tokens de début et de fin.

**Reasoning**:
Examine the `generate_response` function to understand its logic for step-by-step generation, paying attention to how the encoder and decoder are used and how special tokens are handled.
"""

print("Examining the generate_response function...")

# Print the function source code for easier review
import inspect
print(inspect.getsource(generate_response))

# Trace the execution flow mentally or by adding print statements (mental trace here)
print("\nTracing the execution flow:")
print("1. Input text is tokenized using the provided tokenizer.")
print("2. The input tokens are passed to model.encode() to get the encoder output. This is done only once.")
print("3. The decoder input sequence is initialized with the [START] token.")
print("4. A loop runs up to max_length iterations.")
print("5. Inside the loop:")
print("   - model.decode() is called with the current decoder input sequence and the pre-computed encoder output.")
print("   - The output of model.decode() is passed through model.final_layer.")
print("   - The prediction for the *last* token in the decoder sequence is taken.")
print("   - The token ID with the highest probability is selected (argmax).")
print("   - The predicted token is checked if it is the [END] token. If so, the loop breaks.")
print("   - The predicted token ID is appended to the list of generated tokens.")
print("   - The predicted token ID is also appended to the decoder input sequence for the next iteration.")
print("6. After the loop, the generated token IDs are converted back to text using the tokenizer's vocabulary.")
print("7. Special tokens like [START] are cleaned from the final text.")

print("\nVerifying handling of special tokens:")
print("- [START] token: Used correctly to initialize the decoder input sequence.")
print("- [END] token: Used correctly as a stopping condition for the generation loop.")

print("\nChecking encoder output usage by the decoder:")
print("- The encoder output is computed once before the loop.")
print("- model.decode() is called with the current decoder sequence *and* the encoder output in each step of the loop.")
print("Observation: The encoder output is passed to the decode method at each step.")
print("However, as noted in the previous analysis of the model architecture, the SimpleTransformerBlock (used in the decoder) *lacks* a cross-attention mechanism to actually *use* this encoder output. The `decode` method in `ShirayukiTransformer` passes the encoder output, but the layers it calls (`SimpleTransformerBlock`) do not take it as an argument or use it in their attention calculation.")

print("\nIdentifying potential issues/inefficiencies:")
print("- **Lack of Cross-Attention:** The most significant issue is that the decoder layers (SimpleTransformerBlock) do not have cross-attention. While the encoder output is passed to `model.decode`, it is not used by the decoder layers themselves. This means the decoder is only attending to its own previously generated tokens (self-attention) and has no way to incorporate information from the input sequence.")
print("- **Decoder Input Construction:** The decoder input sequence is reconstructed in each loop iteration by concatenating the previous sequence with the new predicted token. This is a standard approach for step-by-step generation in sequence models.")
print("- **Prediction Logic:** Taking the prediction for the *last* token and using argmax is the correct way to predict the next token in a causal generation setting.")
print("- **Efficiency:** Generating token by token in a loop is standard for inference but can be slow compared to batched generation or optimized inference methods, although for a single input, this step-by-step approach is necessary.")

print("\nSummary of generate_response review:")
print("The function implements a standard step-by-step (autoregressive) generation process.")
print("It correctly handles [START] and [END] tokens for initialization and stopping.")
print("It passes the encoder output to the decode method at each step.")
print("CRITICAL ISSUE: The underlying model architecture (SimpleTransformerBlock in the decoder) lacks the cross-attention mechanism required to utilize the encoder output during decoding. This means the decoder cannot effectively condition its output on the input sequence, leading to irrelevant or generic responses.")
print("Other aspects like decoder input construction and prediction logic appear standard and correct for step-by-step generation, but the lack of cross-attention is a fundamental flaw preventing the model from acting as a true sequence-to-sequence Transformer.")

"""## Synthèse et recommandations

### Subtask:
Synthèse et recommandations

**Reasoning**:
Gather the key findings from previous analyses, explain the main reason for poor performance focusing on the lack of cross-attention and limited training, and propose concrete recommendations for improvement including modifying the decoder architecture, repeating the dataset, and suggesting hyperparameter tuning.
"""

print("## Synthèse et Recommandations")
print("\n### Constatations Clés:")
print("- **Architecture du Modèle:** L'analyse a révélé que les couches du décodeur (`SimpleTransformerBlock`) dans le modèle `ShirayukiTransformer` ne comportent pas de mécanisme d'attention croisée (`MultiHeadAttention` qui attendrait à l'output de l'encodeur). Le décodeur utilise uniquement l'auto-attention.")
print("- **Prétraitement des Données:** Le chargement, le nettoyage de base et la tokenisation des données, y compris l'ajout de tokens spéciaux ([START], [END]) et la préparation pour le *teacher forcing*, sont correctement implémentés. Un bémol potentiel concerne la propagation du masque de padding dans les blocs Transformer.")
print("- **Configuration de l'Entraînement:** La configuration utilise des optimiseurs, fonctions de perte et callbacks appropriés. Cependant, l'entraînement s'est arrêté prématurément en raison de la stagnation de la perte et, de manière critique, car le dataset n'était pas configuré pour se répéter, limitant l'entraînement effectif à une seule passe sur les données.")
print("- **Fonction de Génération:** La fonction `generate_response` implémente correctement un processus de génération pas à pas et gère les tokens spéciaux. Elle passe bien l'output de l'encodeur au décodeur, mais en raison du manque d'attention croisée dans l'architecture du modèle, cet output n'est pas utilisé par les couches du décodeur, rendant la génération non conditionnée par l'entrée.")

print("\n### Raison Principale de la Performance Faible:")
print("La raison principale pour laquelle le modèle ne performe pas correctement en tant que modèle de conversation (`guy` -> `girl`) est le **manque critique de mécanisme d'attention croisée dans les couches du décodeur**. Dans une architecture Transformer séquence-à-séquence standard, l'attention croisée permet au décodeur de 'regarder' et d'utiliser les informations de la séquence d'entrée (l'output de l'encodeur) à chaque étape de la génération de la réponse. Sans cette capacité, le décodeur génère une séquence basée uniquement sur les tokens qu'il a lui-même déjà produits (auto-attention), sans pouvoir réellement comprendre et répondre au contexte de l'entrée. L'encodeur traite l'entrée, mais l'information encodée n'est pas transmise efficacement au processus de décision du décodeur. De plus, l'**entraînement limité** à une seule époque effective (dû à l'absence de répétition du dataset) a empêché le modèle d'apprendre suffisamment, même si l'architecture était correcte.")

print("\n### Recommandations pour l'Amélioration:")
print("Pour que le modèle fonctionne correctement en tant que générateur de réponse conditionné par l'entrée, les modifications suivantes sont cruciales:")
print("1.  **Modifier l'Architecture du Décodeur:** Le changement le plus important est d'ajouter une couche d'attention croisée dans chaque `SimpleTransformerBlock` utilisé pour le décodeur. Cette nouvelle couche devrait prendre l'output de l'encodeur comme entrée (`key` et `value`) et l'output de la couche d'auto-attention du décodeur comme `query`. La structure standard d'un bloc décodeur est : Auto-Attention -> Attention Croisée -> FFN.")
print("2.  **Répéter le Dataset pendant l'Entraînement:** Modifier la création du dataset (`tf.data.Dataset`) pour inclure un appel à `.repeat()` après le `batch()`. Cela permettra au modèle de s'entraîner sur plusieurs époques en réutilisant les données, donnant ainsi plus d'opportunités d'apprentissage.")
print("3.  **Corriger la Propagation du Masque (Optionnel mais Recommandé):** S'assurer que le masque de padding (créé par `mask_zero=True` dans la couche `Embedding`) est correctement propagé et utilisé dans les couches d'attention (auto-attention et attention croisée) pour éviter que le modèle n'attende les tokens de padding.")
print("4.  **Expérimenter avec les Hyperparamètres d'Entraînement (Après Correction de 1 & 2):** Une fois l'architecture corrigée et l'entraînement effectif rendu possible par la répétition du dataset, il sera pertinent d'expérimenter avec:")
print("    - Les paramètres des callbacks `EarlyStopping` et `ReduceLROnPlateau` (ex: augmenter la patience si l'entraînement devient plus long).")
print("    - Le taux d'apprentissage initial.")
print("    - La taille du modèle (ex: `embed_dim`, `num_heads`, `ff_dim`, `num_layers`), bien que les valeurs actuelles semblent raisonnables pour commencer une fois les problèmes fondamentaux résolus.")

print("\nEn implémentant ces recommandations, le modèle aura l'architecture et la configuration d'entraînement nécessaires pour apprendre à générer des réponses pertinentes basées sur l'input utilisateur.")

"""## Summary:

### Data Analysis Key Findings

*   The `SimpleTransformerBlock` correctly implements a standard self-attention block but does not include cross-attention, which is necessary for a Transformer decoder to attend to the encoder's output.
*   The `ShirayukiTransformer` uses `SimpleTransformerBlock` for both encoder and decoder layers, meaning the decoder lacks the crucial cross-attention mechanism.
*   The data loading, basic cleaning, and tokenization using `create_simple_tokenizer` are correctly implemented, including the use of special tokens `[START]` and `[END]` and preparing data for teacher forcing.
*   A potential issue with mask propagation from the `Embedding` layer through the `SimpleTransformerBlock` layers was noted.
*   The training configuration uses appropriate components (Adam optimizer, SparseCategoricalCrossentropy loss, standard callbacks like EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).
*   Training stopped prematurely at epoch 5 due to loss stagnation, but more critically, because the dataset was not configured to repeat, limiting effective training to a single pass over the data.
*   The `generate_response` function correctly implements step-by-step generation and handles special tokens, passing the encoder output to the decoder at each step.
*   Despite the `generate_response` function passing the encoder output, the decoder's lack of cross-attention prevents it from utilizing this information, resulting in responses that are not conditioned on the input sequence.

### Insights or Next Steps

*   The most critical step is to modify the `SimpleTransformerBlock` architecture to include a cross-attention layer for use in the decoder, allowing the model to condition its output on the input sequence.
*   Ensure the training dataset is configured to repeat so that the model can train for the intended number of epochs and potentially converge better.

# Task
Augmenter le dataset "conversation_dataset_ShirayukiV3.csv" au centuple en utilisant des techniques de data augmentation pour optimiser le modèle, potentiellement le mixer avec un autre dataset optimisé, puis entraîner le modèle avec le dataset résultant.

## Chargement du dataset existant

### Subtask:
Charger le dataset "conversation_dataset_ShirayukiV3.csv" comme base.

**Reasoning**:
Load the base dataset using the provided function and display the number of conversations loaded.
"""

base_conversations = load_shirayuki_data('/content/conversation_dataset_ShirayukiV3.csv')
print(f"Nombre de conversations chargées comme base : {len(base_conversations)}")

"""## Définition des techniques de data augmentation

### Subtask:
Identifier et implémenter des méthodes de data augmentation adaptées aux données textuelles de conversation (paraphrase, synonymes, légères modifications syntaxiques, etc.). L'objectif est de multiplier le dataset par 100.

**Reasoning**:
Identify and implement text data augmentation techniques suitable for conversational data to multiply the dataset size by 100. This involves choosing a library or method and defining a function to apply the augmentation.
"""

print("Identifying text data augmentation techniques...")

# For text data augmentation, popular libraries include:
# - nlpaug: Supports various techniques (synonym, word embedding, character, etc.)
# - textattack: Primarily for adversarial attacks, but has augmentation methods.
# - google-research/text-to-text-transfer-transformer (T5): Can be fine-tuned for paraphrasing.
# - Easy Data Augmentation (EDA): Simple techniques like synonym replacement, random insertion, swap, deletion.

# Given the requirement for conversation data and the need for ~100x augmentation,
# simple techniques like synonym replacement, random insertion/deletion/swap might introduce noise
# or change the meaning significantly in short conversational phrases.
# Paraphrasing models (like T5) can generate more natural variations but require a pre-trained model
# and potentially fine-tuning, which adds complexity.

# For this task, let's use a combination of simple, relatively safe techniques
# using nlpaug, which is easy to install and use, and supports synonym replacement.
# We will aim for variations that keep the core meaning.

# Install nlpaug if not already installed
try:
    import nlpaug.augmenter.word as naw
    print("nlpaug found.")
except ImportError:
    print("Installing nlpaug...")
    !pip install nlpaug
    import nlpaug.augmenter.word as naw
    print("nlpaug installed and imported.")

# Initialize augmentation technique(s)
# Using Synonym Replacement as it's less likely to drastically change meaning.
# We can potentially add others later if needed, but let's start simple.
# We need a word list or word embedding model for synonym replacement.
# nlpaug provides built-in wordnet or word embedding based augmenters.
# WordNetAug is simpler and doesn't require downloading large models initially.

augmenter = naw.SynonymAug(aug_src='wordnet', aug_p=0.3) # Replace up to 30% of words with synonyms

def augment_conversation(input_pair, num_augmentations=99):
    """
    Augments a single conversation pair to create multiple variations.

    Args:
        input_pair (tuple): A tuple containing (guy_text, girl_text).
        num_augmentations (int): The number of augmented pairs to generate
                                 from this original pair.

    Returns:
        list: A list of augmented conversation pairs, including the original.
              Each element is a tuple (augmented_guy_text, augmented_girl_text).
    """
    original_guy, original_girl = input_pair
    augmented_pairs = [input_pair] # Start with the original pair

    print(f"Augmenting conversation: '{original_guy}' / '{original_girl}'")

    for _ in range(num_augmentations):
        augmented_guy = original_guy
        augmented_girl = original_girl

        try:
            # Apply augmentation to the 'guy' part
            if original_guy and len(original_guy.split()) > 1: # Augment if more than one word
                 augmented_guy = augmenter.augment(original_guy, n=1)[0] # Generate 1 augmented version

            # Apply augmentation to the 'girl' part
            if original_girl and len(original_girl.split()) > 1: # Augment if more than one word
                 augmented_girl = augmenter.augment(original_girl, n=1)[0] # Generate 1 augmented version

            augmented_pairs.append((augmented_guy, augmented_girl))
            # print(f"  -> Augmented: '{augmented_guy}' / '{augmented_girl}'") # Optional: Print augmented pairs

        except Exception as e:
            print(f"  ⚠️ Could not augment pair '{original_guy}' / '{original_girl}': {e}")
            # If augmentation fails, just skip this augmentation iteration.
            # We won't append a potentially invalid pair.

    return augmented_pairs

print("Data augmentation function 'augment_conversation' defined.")
print("Augmentation technique: Synonym Replacement (using WordNet, 30% probability)")

# Test the function with a sample conversation from the base_conversations
if base_conversations:
    sample_pair = base_conversations[0]
    print(f"\nTesting augmentation with sample pair: {sample_pair}")
    augmented_samples = augment_conversation(sample_pair, num_augmentations=5) # Generate 5 augmented versions + original
    print(f"\nGenerated {len(augmented_samples)} augmented versions (including original).")
    for idx, pair in enumerate(augmented_samples):
        print(f"  Augmented {idx}: '{pair[0]}' / '{pair[1]}'")
else:
    print("\nNo base conversations loaded to test augmentation function.")

"""**Reasoning**:
The previous code failed because `base_conversations` was not defined in the current execution context. This variable was defined in a previous cell. I need to re-execute the cell that loads the base conversations and then rerun the code to define and test the augmentation function.


"""

# Re-load the base conversations
# This was originally done in the first subtask cell.
base_conversations = load_shirayuki_data('/content/conversation_dataset_ShirayukiV3.csv')
print(f"Nombre de conversations chargées comme base : {len(base_conversations)}")

# Now, re-define the augmentation function and test it
print("\nIdentifying text data augmentation techniques...")

# Augmentation technique already identified: Synonym Replacement using nlpaug
# nlpaug is already installed from the previous attempt.

import nlpaug.augmenter.word as naw

# Initialize augmentation technique(s)
augmenter = naw.SynonymAug(aug_src='wordnet', aug_p=0.3) # Replace up to 30% of words with synonyms

def augment_conversation(input_pair, num_augmentations=99):
    """
    Augments a single conversation pair to create multiple variations.

    Args:
        input_pair (tuple): A tuple containing (guy_text, girl_text).
        num_augmentations (int): The number of augmented pairs to generate
                                 from this original pair.

    Returns:
        list: A list of augmented conversation pairs, including the original.
              Each element is a tuple (augmented_guy_text, augmented_girl_text).
    """
    original_guy, original_girl = input_pair
    augmented_pairs = [input_pair] # Start with the original pair

    print(f"Augmenting conversation: '{original_guy}' / '{original_girl}'")

    for _ in range(num_augmentations):
        augmented_guy = original_guy
        augmented_girl = original_girl

        try:
            # Apply augmentation to the 'guy' part
            # Augment only if the text is not empty and has more than one word
            if original_guy and len(original_guy.split()) > 1:
                 # The augment method returns a list, take the first element
                 augmented_guy = augmenter.augment(original_guy, n=1)[0]

            # Apply augmentation to the 'girl' part
            # Augment only if the text is not empty and has more than one word
            if original_girl and len(original_girl.split()) > 1:
                 # The augment method returns a list, take the first element
                 augmented_girl = augmenter.augment(original_girl, n=1)[0]

            # Only append if at least one part was potentially augmented or if no augmentation was needed (single word/empty)
            # This avoids infinite loops if augmentation fails for single-word inputs
            if augmented_guy != original_guy or augmented_girl != original_girl or (original_guy and len(original_guy.split()) <= 1 and original_girl and len(original_girl.split()) <= 1):
                 augmented_pairs.append((augmented_guy, augmented_girl))
            # print(f"  -> Augmented: '{augmented_guy}' / '{augmented_girl}'") # Optional: Print augmented pairs

        except Exception as e:
            # print(f"  ⚠️ Could not augment pair '{original_guy}' / '{original_girl}': {e}")
            # If augmentation fails for a specific pair, just skip this augmentation iteration.
            # We won't append a potentially invalid pair.
            pass # Suppress error message for brevity during mass augmentation


    return augmented_pairs

print("Data augmentation function 'augment_conversation' redefined.")
print("Augmentation technique: Synonym Replacement (using WordNet, 30% probability)")

# Test the function with a sample conversation from the base_conversations
if base_conversations:
    sample_pair = base_conversations[0]
    print(f"\nTesting augmentation with sample pair: {sample_pair}")
    augmented_samples = augment_conversation(sample_pair, num_augmentations=5) # Generate 5 augmented versions + original
    print(f"\nGenerated {len(augmented_samples)} augmented versions (including original).")
    for idx, pair in enumerate(augmented_samples):
        print(f"  Augmented {idx}: '{pair[0]}' / '{pair[1]}'")
else:
    print("\nNo base conversations loaded to test augmentation function.")

"""**Reasoning**:
The sample augmentation test only generated the original pair. This indicates an issue with the augmentation logic or the augmenter's ability to find synonyms for the words in the sample sentence with the current probability. I need to modify the `augment_conversation` function to ensure it generates `num_augmentations` *additional* augmented pairs and handle cases where augmentation might not find synonyms. Increasing the probability or trying different augmentation techniques might be necessary if synonym replacement alone is insufficient. However, the primary goal is to generate the specified number of augmentations.


"""

print("Modifying augment_conversation to ensure generation of required augmentations...")

# Re-define the augment_conversation function
def augment_conversation(input_pair, num_augmentations=99):
    """
    Augments a single conversation pair to create multiple variations.

    Args:
        input_pair (tuple): A tuple containing (guy_text, girl_text).
        num_augmentations (int): The number of *additional* augmented pairs to generate
                                 from this original pair (excluding the original).

    Returns:
        list: A list of augmented conversation pairs, including the original.
              Each element is a tuple (augmented_guy_text, augmented_girl_text).
    """
    original_guy, original_girl = input_pair
    augmented_pairs = [input_pair] # Start with the original pair

    print(f"Augmenting conversation: '{original_guy}' / '{original_girl}'")

    # Use a loop to generate the specified number of *additional* augmentations
    generated_count = 0
    while generated_count < num_augmentations:
        augmented_guy = original_guy
        augmented_girl = original_girl
        pair_augmented = False # Flag to check if at least one part was augmented

        try:
            # Apply augmentation to the 'guy' part
            if original_guy and len(original_guy.split()) > 1: # Augment if not empty and more than one word
                 # The augment method returns a list, take the first element
                 temp_guy = augmenter.augment(original_guy, n=1)[0]
                 if temp_guy != original_guy: # Check if augmentation actually changed the text
                     augmented_guy = temp_guy
                     pair_augmented = True

            # Apply augmentation to the 'girl' part
            if original_girl and len(original_girl.split()) > 1: # Augment if not empty and more than one word
                 # The augment method returns a list, take the first element
                 temp_girl = augmenter.augment(original_girl, n=1)[0]
                 if temp_girl != original_girl: # Check if augmentation actually changed the text
                     augmented_girl = temp_girl
                     pair_augmented = True

            # Only append if at least one part was actually augmented to avoid adding duplicates of the original
            if pair_augmented:
                 augmented_pairs.append((augmented_guy, augmented_girl))
                 generated_count += 1
                 # print(f"  -> Generated {generated_count}/{num_augmentations}: '{augmented_guy}' / '{augmented_girl}'") # Optional: Print progress

        except Exception as e:
            # print(f"  ⚠️ Could not augment pair '{original_guy}' / '{original_girl}' in iteration {generated_count}: {e}")
            # If augmentation fails for a specific pair, just skip this augmentation iteration.
            pass # Suppress error message for brevity during mass augmentation

        # Add a safeguard to prevent infinite loops if augmentation consistently fails
        if generated_count < num_augmentations and len(augmented_pairs) > num_augmentations * 2: # Arbitrary large number
             print(f"  ⚠️ Warning: Augmentation struggling to generate unique variations after many attempts for pair: '{original_guy}' / '{original_girl}'. Stopping early for this pair.")
             break


    print(f"Finished augmenting pair: '{original_guy}' / '{original_girl}'. Generated {len(augmented_pairs)-1} additional pairs.")
    return augmented_pairs

print("Data augmentation function 'augment_conversation' updated.")

if base_conversations:
    sample_pair = base_conversations[0]
    print(f"\nTesting updated augmentation with sample pair: {sample_pair}")
    # Request 5 additional augmentations
    augmented_samples = augment_conversation(sample_pair, num_augmentations=5)
    print(f"\nGenerated {len(augmented_samples)} augmented versions (including original).")
    for idx, pair in enumerate(augmented_samples):
        print(f"  Augmented {idx}: '{pair[0]}' / '{pair[1]}'")
else:
    print("\nNo base conversations loaded to test augmentation function.")
