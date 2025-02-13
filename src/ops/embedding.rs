

use std::fs::File;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::io::{BufReader, BufWriter, BufRead, Read, Seek, SeekFrom};
use std::collections::{HashMap, HashSet};

use num_cpus;
use fastrand;
use serde_json;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use indicatif::{ProgressBar, ProgressStyle};

use tracing::{debug, info, error};


/// Default paths for the model.
const DEFAULT_MODEL_PATH: &str = "models/word2vec_model.json";
/// Default path for the corpus.
const DEFAULT_CORPUS_PATH: &str = "data/word2vec_corpus.txt";

#[derive(Serialize, Deserialize)]
/// Structure for Word2Vec model.
pub struct SingleThreadWord2Vec {
    vocab: HashMap<String, usize>,        // Word to index mapping
    index_to_word: Vec<String>,           // Index to word mapping
    input_vectors: Vec<Vec<f32>>,         // Input layer embeddings
    output_vectors: Vec<Vec<f32>>,        // Output layer embeddings
    window_size: usize,                   // Context window size
    embedding_dim: usize,                 // Embedding dimension
    negative_samples: usize,              // Negative samples per positive sample
}

impl SingleThreadWord2Vec {
    /// Initialize the Word2Vec model.
    pub fn new(vocab_size: usize, embedding_dim: usize, window_size: usize, negative_samples: usize) -> Self {
        SingleThreadWord2Vec {
            vocab: HashMap::new(),
            index_to_word: Vec::new(),
            input_vectors: (0..vocab_size)
                .into_par_iter()
                .map(|_| (0..embedding_dim).map(|_| fastrand::f32() - 0.5).collect())
                .collect(),
            output_vectors: (0..vocab_size)
                .into_par_iter()
                .map(|_| (0..embedding_dim).map(|_| fastrand::f32() - 0.5).collect())
                .collect(),
            window_size,
            embedding_dim,
            negative_samples,
        }
    }

    /// Read the corpus from a file.
    pub fn read_corpus(path: Option<&str>) -> Result<Vec<String>, Box<dyn Error>> {
        let path = path.unwrap_or(DEFAULT_CORPUS_PATH);
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(reader.lines().collect::<Result<Vec<_>, _>>()?)
    }

    /// Load the model from a file.
    pub fn load(path: Option<&str>) -> Result<Self, Box<dyn Error>> {
        let path = path.unwrap_or(DEFAULT_MODEL_PATH);
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(serde_json::from_reader(reader)?)
    }

    /// Train the model using Skip-Gram with Negative Sampling.
    pub fn train(&mut self, corpus: &[String], epochs: usize, learning_rate: f32) {
        let pb = ProgressBar::new((epochs * corpus.len()) as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap());

        for _epoch in 0..epochs {
            for (idx, word) in corpus.iter().enumerate() {
                if let Some(&word_idx) = self.vocab.get(word) {
                    let start = if idx >= self.window_size { idx - self.window_size } else { 0 };
                    let end = usize::min(idx + self.window_size + 1, corpus.len());

                    for context_word in &corpus[start..end] {
                        if context_word == word { continue; }
                        if let Some(&context_idx) = self.vocab.get(context_word) {
                            self.update_embeddings(word_idx, context_idx, 1.0, learning_rate);

                            let negative_samples: HashSet<usize> = (0..self.negative_samples)
                                .map(|_| fastrand::usize(0..self.vocab.len()))
                                .filter(|&idx| idx != context_idx)
                                .collect();

                            for &neg_idx in &negative_samples {
                                self.update_embeddings(word_idx, neg_idx, 0.0, learning_rate);
                            }
                        }
                    }
                }
                pb.inc(1);
            }
            pb.set_message(format!("Epoch {}/{}", _epoch + 1, epochs));
        }
        pb.finish_with_message("Training complete");
    }

    /// Update embeddings using gradient descent.
    fn update_embeddings(&mut self, target_idx: usize, context_idx: usize, label: f32, learning_rate: f32) {
        let input_vec = self.input_vectors[target_idx].clone();
        let output_vec = self.output_vectors[context_idx].clone();

        let dot_product: f32 = input_vec.iter().zip(output_vec.iter()).map(|(x, y)| x * y).sum();
        let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
        let error = sigmoid - label;

        // Sequential updates
        for i in 0..self.embedding_dim {
            self.input_vectors[target_idx][i] -= learning_rate * error * output_vec[i];
            self.output_vectors[context_idx][i] -= learning_rate * error * input_vec[i];
        }
    }

    /// Save the model to a file.
    pub fn save(&self, path: Option<&str>) -> Result<(), Box<dyn Error>> {
        let path = path.unwrap_or(DEFAULT_MODEL_PATH);
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &self)?;
        Ok(())
    }
}

/// Visitor for deserialization.
struct VisitorImpl<'a> {
    vocab: &'a mut Option<HashMap<String, usize>>,
    index_to_word: &'a mut Option<Vec<String>>,
}

impl<'de, 'a> serde::de::Visitor<'de> for VisitorImpl<'a> {
    type Value = ();

    /// Get the expected type.
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a Word2Vec model")
    }

    /// Visit the map.
    fn visit_map<M>(self, mut map: M) -> Result<(), M::Error>
    where
        M: serde::de::MapAccess<'de>,
    {
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "vocab" => *self.vocab = Some(map.next_value()?),
                "index_to_word" => *self.index_to_word = Some(map.next_value()?),
                _ => { let _ = map.next_value::<serde::de::IgnoredAny>()?; }
            }
        }
        Ok(())
    }
}

#[derive(Serialize)]
/// Helper struct for serialization.
struct SerializableWord2Vec<'a> {
    vocab: &'a HashMap<String, usize>,
    index_to_word: &'a Vec<String>,
    input_vectors: &'a Vec<Vec<f32>>,
    output_vectors: &'a Vec<Vec<f32>>,
    window_size: usize,
    embedding_dim: usize,
    negative_samples: usize,
}

/// Structure for Word2Vec model.
pub struct MultiThreadWord2Vec {
    vocab: HashMap<String, usize>,          // Word to index mapping
    index_to_word: Vec<String>,            // Index to word mapping
    input_vectors: Arc<Mutex<Vec<Vec<f32>>>>, // Input layer embeddings
    output_vectors: Arc<Mutex<Vec<Vec<f32>>>>, // Output layer embeddings
    window_size: usize,                    // Context window size
    embedding_dim: usize,                  // Embedding dimension
    negative_samples: usize,               // Negative samples per positive sample
}

impl Serialize for MultiThreadWord2Vec {
    /// Serialize the model to a JSON file.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("Word2Vec", 7)?;
        state.serialize_field("vocab", &self.vocab)?;
        state.serialize_field("index_to_word", &self.index_to_word)?;
        state.serialize_field("input_vectors", &*self.input_vectors.lock().unwrap())?;
        state.serialize_field("output_vectors", &*self.output_vectors.lock().unwrap())?;
        state.serialize_field("window_size", &self.window_size)?;
        state.serialize_field("embedding_dim", &self.embedding_dim)?;
        state.serialize_field("negative_samples", &self.negative_samples)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for MultiThreadWord2Vec {
    /// Deserialize the model from a JSON file.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::MapAccess;
        
        let mut vocab = None;
        let mut index_to_word = None;
        
        let mut map = deserializer.deserialize_struct("Word2Vec", 
            &["vocab", "index_to_word"], 
            VisitorImpl { 
                vocab: &mut vocab,
                index_to_word: &mut index_to_word,
            }
        )?;

        let vocab = vocab.ok_or_else(|| serde::de::Error::missing_field("vocab"))?;
        let index_to_word = index_to_word.ok_or_else(|| serde::de::Error::missing_field("index_to_word"))?;
        let vocab_size = vocab.len();
        let embedding_dim = 100; // Default value
        let window_size = 5; // Default value  
        let negative_samples = 5; // Default value

        Ok(MultiThreadWord2Vec {
            vocab,
            index_to_word,
            input_vectors: Arc::new(Mutex::new(
                (0..vocab_size)
                    .map(|_| (0..embedding_dim).map(|_| fastrand::f32() - 0.5).collect())
                    .collect(),
            )),
            output_vectors: Arc::new(Mutex::new(
                (0..vocab_size)
                    .map(|_| (0..embedding_dim).map(|_| fastrand::f32() - 0.5).collect())
                    .collect(),
            )),
            window_size,
            embedding_dim,
            negative_samples,
        })
    }
}

impl MultiThreadWord2Vec {
    /// Initialize the Word2Vec model.
    pub fn new(vocab_size: usize, embedding_dim: usize, window_size: usize, negative_samples: usize) -> Self {
        MultiThreadWord2Vec {
            vocab: HashMap::new(),
            index_to_word: Vec::new(),
            input_vectors: Arc::new(Mutex::new(
                (0..vocab_size)
                    .map(|_| (0..embedding_dim).map(|_| fastrand::f32() - 0.5).collect())
                    .collect(),
            )),
            output_vectors: Arc::new(Mutex::new(
                (0..vocab_size)
                    .map(|_| (0..embedding_dim).map(|_| fastrand::f32() - 0.5).collect())
                    .collect(),
            )),
            window_size,
            embedding_dim,
            negative_samples,
        }
        
    }

    /// Read the corpus from a file.
    pub fn read_corpus(path: Option<&str>) -> Result<Vec<String>, Box<dyn Error>> {
        let path = path.unwrap_or(DEFAULT_CORPUS_PATH);
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(reader.lines().collect::<Result<Vec<_>, _>>()?)
    }

    /// Load the model from a file.
    pub fn load(path: Option<&str>) -> Result<Self, Box<dyn Error>> {
        let path = path.unwrap_or(DEFAULT_MODEL_PATH);
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(serde_json::from_reader(reader)?)
    }

    /// Train the model using Skip-Gram with Negative Sampling.
    pub fn train(&mut self, corpus: &[String], epochs: usize, learning_rate: f32) {
        let pb = ProgressBar::new((epochs * corpus.len()) as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap());

        for _epoch in 0..epochs {
            corpus.par_iter().enumerate().for_each(|(idx, word)| {
                if let Some(&word_idx) = self.vocab.get(word) {
                    let start = if idx >= self.window_size { idx - self.window_size } else { 0 };
                    let end = usize::min(idx + self.window_size + 1, corpus.len());
            
                    let mut local_updates = vec![];
                    for context_word in &corpus[start..end] {
                        if context_word == word { continue; }
                        if let Some(&context_idx) = self.vocab.get(context_word) {
                            local_updates.push((word_idx, context_idx, 1.0));
                            let negative_samples: HashSet<usize> = (0..self.negative_samples)
                                .map(|_| fastrand::usize(0..self.vocab.len()))
                                .filter(|&idx| idx != context_idx)
                                .collect();
            
                            for &neg_idx in &negative_samples {
                                local_updates.push((word_idx, neg_idx, 0.0));
                            }
                        }
                    }
            
                    // Apply updates sequentially
                    for (target_idx, context_idx, label) in local_updates {
                        self.update_embeddings(target_idx, context_idx, label, learning_rate);
                    }
                }
                pb.inc(1);
            });
            pb.set_message(format!("Epoch {}/{}", _epoch + 1, epochs));
        }
        pb.finish_with_message("Training complete");
    }

    /// Update embeddings using `gradient descent`.
    fn update_embeddings(&self, target_idx: usize, context_idx: usize, label: f32, learning_rate: f32) {
        // Get input vectors - handle potential poison error
        let input_vec = match self.input_vectors.lock() {
            Ok(guard) => guard[target_idx].clone(),
            Err(poisoned) => poisoned.into_inner()[target_idx].clone(),
        };

        // Get output vectors - handle potential poison error
        let output_vec = match self.output_vectors.lock() {
            Ok(guard) => guard[context_idx].clone(),
            Err(poisoned) => poisoned.into_inner()[context_idx].clone(),
        };

        let dot_product: f32 = input_vec.iter().zip(output_vec.iter()).map(|(x, y)| x * y).sum();
        let sigmoid = 1.0 / (1.0 + (-dot_product).exp());
        let error = sigmoid - label;

        // Update vectors with error handling
        if let Ok(mut input_vectors) = self.input_vectors.lock() {
            for i in 0..self.embedding_dim {
                input_vectors[target_idx][i] -= learning_rate * error * output_vec[i];
            }
        }

        if let Ok(mut output_vectors) = self.output_vectors.lock() {
            for i in 0..self.embedding_dim {
                output_vectors[context_idx][i] -= learning_rate * error * input_vec[i];
            }
        }
    }
    
    /// Save the model to a file.
    pub fn save(&self, path: Option<&str>) -> Result<(), Box<dyn Error>> {
        let path = path.unwrap_or(DEFAULT_MODEL_PATH);
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Handle potential poison errors during serialization
        let input_vectors = self.input_vectors.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let output_vectors = self.output_vectors.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        
        let serializable_model = SerializableWord2Vec {
            vocab: &self.vocab,
            index_to_word: &self.index_to_word,
            input_vectors: &*input_vectors,
            output_vectors: &*output_vectors,
            window_size: self.window_size,
            embedding_dim: self.embedding_dim,
            negative_samples: self.negative_samples,
        };
        
        serde_json::to_writer(&mut writer, &serializable_model)?;
        Ok(())
    }
}


#[derive(Serialize, Deserialize)]
/// A struct to load a Word2Vec model from a file.
pub struct Word2VecFromFile{
    vocab: HashMap<String, usize>,        // Word to index mapping
    index_to_word: Vec<String>,           // Index to word mapping
    input_vectors: Vec<Vec<f32>>,         // Input layer embeddings
    output_vectors: Vec<Vec<f32>>,        // Output layer embeddings
    window_size: usize,                   // Context window size
    embedding_dim: usize,                 // Embedding dimension
    negative_samples: usize,              // Negative samples per positive sample
}

impl Word2VecFromFile {
    /// Read the JSON file in parallel.
    pub fn read_json_parallel(path: &str) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();
        let chunk_size = file_size / num_cpus::get() as u64;
        
        let chunks: Vec<String> = (0..num_cpus::get() as u64)
            .into_par_iter()
            .map(|i| {
                let mut reader = BufReader::new(File::open(path).ok()?);
                let start = i * chunk_size;
                let end = if i == num_cpus::get() as u64 - 1 {
                    file_size
                } else {
                    (i + 1) * chunk_size
                };

                reader.seek(SeekFrom::Start(start)).ok()?;
                let mut buffer = vec![0; (end - start) as usize];
                reader.read_exact(&mut buffer).ok()?;

                String::from_utf8(buffer).ok()
            })
            .collect::<Option<Vec<_>>>()
            .ok_or("Failed to read file chunks")?;

        // Combine chunks and parse JSON
        let combined = chunks.join("");
        Ok(serde_json::from_str(&combined)?)
    }

    /// Create a new `Word2VecFromFile` instance.
    pub fn new(path: &str) -> Result<Self, Box<dyn Error>> {
        Self::read_json_parallel(path)
    }

    /// Get the embedding for a word.
    pub fn get_embedding(&self, word: &str) -> Option<&[f32]> {
        let idx = self.vocab.get(word)?;
        Some(&self.input_vectors[*idx])
    }

    /// Get the embedding for a text by averaging the embeddings of its words.
    pub fn get_embedding_for_text(&self, text: &str) -> Option<Vec<f32>> {
        debug!("Getting embedding for text: {}.", text);
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut embeddings: Vec<Vec<f32>> = Vec::new();
        
        for word in words {
            if let Some(embedding) = self.get_embedding(word) {
                embeddings.push(embedding.to_vec());
            }
        }

        if embeddings.is_empty() {
            return None; // No embeddings found for any words
        }

        // Average the embeddings
        let embedding_dim = embeddings[0].len();
        let mut averaged_embedding = vec![0.0; embedding_dim];

        for embedding in embeddings.iter() {
            for (i, &value) in embedding.iter().enumerate() {
                averaged_embedding[i] += value;
            }
        }

        // Divide by the number of embeddings to get the average
        let count = embeddings.len() as f32;
        for value in &mut averaged_embedding {
            *value /= count;
        }

        Some(averaged_embedding)
    }
}
