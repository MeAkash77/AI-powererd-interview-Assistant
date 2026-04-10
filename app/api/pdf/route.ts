import { NextRequest, NextResponse } from "next/server";
import { Pinecone } from '@pinecone-database/pinecone';
import { safePdfParse } from '../../../lib/safePdfParse';

// Use Node.js runtime for PDF processing
export const runtime = 'nodejs';

async function uploadPDFToVectorDB(file: File): Promise<boolean> {
  try {
    if (!process.env.PINECONE_API_KEY) {
      console.warn('Pinecone API key not configured');
      return false;
    }

    if (!process.env.HUGGINGFACE_API_KEY) {
      console.warn('Hugging Face API key not configured');
      return false;
    }

    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });

    // Parse PDF
    const arrayBuffer = await file.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    const pdfData = await safePdfParse(buffer);

    // Split into chunks
    const chunks = splitIntoChunks(pdfData.text, 1000);
    
    // Generate embeddings and store
    const indexName = process.env.PINECONE_INDEX_NAME || 'interview-docs';
    const index = pinecone.index(indexName);
    const vectors = [];

    console.log('🧮 Generating Hugging Face embeddings (BAAI/bge-small-en-v1.5)...');
    
    for (let i = 0; i < chunks.length; i++) {
      // Generate embedding using BAAI/bge-small-en-v1.5 model
      const response = await fetch(
        "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5",
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            inputs: chunks[i],
          }),
        }
      );

      // ✅ handle non-JSON safely
      const textResponse = await response.text();

      let result;
      try {
        result = JSON.parse(textResponse);
      } catch (err) {
        console.error(`Raw HF response for chunk ${i}:`, textResponse);
        throw new Error(`Invalid JSON from HuggingFace for chunk ${i}`);
      }

      console.log(`HF Response for chunk ${i}:`, result); // 👈 DEBUG

      // ✅ Flexible embedding validation - handles both 1D and 2D arrays
      let embedding;

      if (Array.isArray(result) && typeof result[0] === "number") {
        // ✅ Case 1: Direct 1D embedding
        embedding = result;
      } else if (Array.isArray(result) && Array.isArray(result[0])) {
        // ✅ Case 2: 2D embedding
        embedding = result[0];
      } else {
        console.error("Invalid embedding format:", result);
        throw new Error(result.error || `Invalid embedding format for chunk ${i}`);
      }
      
      // Validate embedding length
      if (!embedding || !Array.isArray(embedding) || embedding.length === 0) {
        throw new Error(`Empty embedding for chunk ${i}`);
      }

      // ✅ Each vector MUST have id, values, and metadata
      vectors.push({
        id: `${file.name}-chunk-${i}`, // ✅ REQUIRED: unique ID
        values: embedding, // ✅ REQUIRED: array of numbers (384 dimensions)
        metadata: { // ✅ Optional but recommended
          content: chunks[i].slice(0, 1000), // Limit metadata size
          filename: file.name,
          chunkIndex: i,
          title: file.name.replace('.pdf', ''),
          uploadDate: new Date().toISOString(),
          contentType: 'text'
        }
      });
      
      // Log progress every 10 chunks
      if ((i + 1) % 10 === 0) {
        console.log(`📝 Processed ${i + 1}/${chunks.length} chunks`);
      }
      
      // Add a small delay to avoid rate limiting
      if ((i + 1) % 5 === 0) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }

    // ✅ FIX: Use smaller batch size (50) to avoid 502 errors
    const batchSize = 50;
    console.log(`📦 Uploading ${vectors.length} vectors in batches of ${batchSize}...`);
    
    for (let i = 0; i < vectors.length; i += batchSize) {
      const batch = vectors.slice(i, i + batchSize);
      
      // ✅ Ensure each vector in batch has required fields
      const validBatch = batch.filter(v => v.id && v.values && v.values.length === 384);
      
      if (validBatch.length !== batch.length) {
        console.warn(`⚠️ Filtered out ${batch.length - validBatch.length} invalid vectors`);
      }
      
      if (validBatch.length > 0) {
        await index.upsert(validBatch);
        console.log(`✅ Uploaded batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(vectors.length / batchSize)} (${validBatch.length} vectors)`);
      }
      
      // Add small delay between batches
      if (i + batchSize < vectors.length) {
        await new Promise(resolve => setTimeout(resolve, 200));
      }
    }
    
    console.log(`🎉 Successfully uploaded ${vectors.length} chunks for ${file.name}`);
    return true;
  } catch (error) {
    console.error('Error uploading PDF:', error);
    return false;
  }
}

function splitIntoChunks(text: string, chunkSize: number): string[] {
  const words = text.split(' ');
  const chunks: string[] = [];
  
  for (let i = 0; i < words.length; i += chunkSize) {
    chunks.push(words.slice(i, i + chunkSize).join(' '));
  }
  
  return chunks;
}

async function getEmbeddingDimension(): Promise<number> {
  try {
    if (!process.env.HUGGINGFACE_API_KEY) {
      return 384; // Default dimension for bge-small-en-v1.5 is 384
    }
    
    const response = await fetch(
      "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          inputs: "test",
        }),
      }
    );

    // ✅ handle non-JSON safely for dimension check
    const textResponse = await response.text();

    let result;
    try {
      result = JSON.parse(textResponse);
    } catch (err) {
      console.error("Raw HF response for dimension:", textResponse);
      return 384; // Default dimension on parse error
    }
    
    // Flexible dimension extraction
    let embedding;
    if (Array.isArray(result) && typeof result[0] === "number") {
      embedding = result;
    } else if (Array.isArray(result) && Array.isArray(result[0])) {
      embedding = result[0];
    }
    
    if (embedding && Array.isArray(embedding)) {
      console.log(`📏 Detected embedding dimension: ${embedding.length}`);
      return embedding.length;
    }
    
    return 384; // Default dimension for bge-small-en-v1.5
  } catch (error) {
    console.error("Error getting embedding dimension:", error);
    return 384; // Default dimension for bge-small-en-v1.5
  }
}

async function deletePDFFromVectorDB(filename: string): Promise<boolean> {
  try {
    if (!process.env.PINECONE_API_KEY) {
      return false;
    }

    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY,
    });

    const indexName = process.env.PINECONE_INDEX_NAME || 'interview-docs';
    const index = pinecone.index(indexName);
    
    // Get the actual embedding dimension (384 for bge-small-en-v1.5)
    const dimension = await getEmbeddingDimension();
    
    // Find all chunks for this document
    const searchResponse = await index.query({
      vector: new Array(dimension).fill(0),
      topK: 10000,
      includeMetadata: true,
      filter: { filename: { $eq: filename } }
    });

    if (searchResponse.matches && searchResponse.matches.length > 0) {
      const idsToDelete = searchResponse.matches.map(match => match.id);
      
      // Delete in batches to avoid overwhelming the API
      const deleteBatchSize = 100;
      for (let i = 0; i < idsToDelete.length; i += deleteBatchSize) {
        const batch = idsToDelete.slice(i, i + deleteBatchSize);
        await index.deleteMany(batch);
        console.log(`🗑️ Deleted batch ${Math.floor(i / deleteBatchSize) + 1}/${Math.ceil(idsToDelete.length / deleteBatchSize)}`);
      }
      
      console.log(`🗑️ Deleted ${idsToDelete.length} chunks for ${filename}`);
    } else {
      console.log(`ℹ️ No chunks found for ${filename}`);
    }

    return true;
  } catch (error) {
    console.error('Error deleting document:', error);
    return false;
  }
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    if (file.type !== "application/pdf") {
      return NextResponse.json({ error: "Only PDF files are allowed" }, { status: 400 });
    }

    const success = await uploadPDFToVectorDB(file);

    if (success) {
      return NextResponse.json({ 
        message: "PDF uploaded successfully",
        filename: file.name,
        size: file.size
      });
    } else {
      return NextResponse.json({ error: "Failed to upload PDF" }, { status: 500 });
    }
  } catch (error) {
    console.error("PDF upload error:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const filename = searchParams.get("filename");

    if (!filename) {
      return NextResponse.json({ error: "No filename provided" }, { status: 400 });
    }

    const success = await deletePDFFromVectorDB(filename);

    if (success) {
      return NextResponse.json({ message: "PDF deleted successfully" });
    } else {
      return NextResponse.json({ error: "Failed to delete PDF" }, { status: 500 });
    }
  } catch (error) {
    console.error("PDF delete error:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}