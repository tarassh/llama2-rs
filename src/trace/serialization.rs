use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

use super::ExecutionTrace;

pub fn save_trace<P: AsRef<Path>>(trace: &ExecutionTrace, path: P) -> io::Result<()> {
    let encoded = bincode::serialize(trace)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    let mut file = File::create(path)?;
    file.write_all(&encoded)?;
    Ok(())
}

pub fn load_trace<P: AsRef<Path>>(path: P) -> io::Result<ExecutionTrace> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let trace = bincode::deserialize(&buffer)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;

    Ok(trace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::{ExecutionTrace, TraceMetadata};
    use std::fs;

    #[test]
    fn test_save_and_load_trace() {
        let temp_path = "/tmp/test_trace.bin";

        // Create a trace
        let metadata = TraceMetadata {
            model_path: "test_model.bin".to_string(),
            prompt: "test prompt".to_string(),
            steps: 10,
        };
        let mut trace = ExecutionTrace::new(metadata);

        // Add some entries
        let logits = vec![1.0, 2.0, 3.0];
        for i in 0..10 {
            trace.record_step(i, i * 2, &logits);
        }
        trace.finalize();

        // Save and load
        save_trace(&trace, temp_path).unwrap();
        let loaded = load_trace(temp_path).unwrap();

        // Verify
        assert_eq!(trace.root(), loaded.root());
        assert_eq!(trace.entry_count(), loaded.entry_count());

        // Cleanup
        fs::remove_file(temp_path).ok();
    }
}
