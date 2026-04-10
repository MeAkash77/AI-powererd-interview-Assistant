import { Pinecone } from '@pinecone-database/pinecone';
import { safePdfParse } from '../safePdfParse';

export interface PDFDocument {
  id: string;
  title: string;
  content: string;
  chunks: string[];
  metadata: {
    filename: string;
    uploadDate: string;
    pageCount: number;
    contentType: 'text' | 'multimodal';
  };
}

export interface SearchResult {
  content: string;
  score: number;
  metadata: any;
  source: string;
  contentType?: 'text' | 'image' | 'multimodal';
  page?: number;
  startPage?: number;
  endPage?: number;
}

class PineconeService {
  private pinecone: Pinecone | null = null;
  private indexName: string;

  constructor() {
    this.indexName = process.env.PINECONE_INDEX_NAME || 'interview-docs';
    this.initialize();
  }

  private async initialize() {
    try {
      console.log('🔧 Initializing Pinecone service with Hugging Face embeddings...');
      console.log(`📌 Using Pinecone index: ${this.indexName}`);
      
      if (process.env.PINECONE_API_KEY) {
        this.pinecone = new Pinecone({
          apiKey: process.env.PINECONE_API_KEY,
        });
        console.log('✅ Pinecone client initialized');
        
        // Verify index exists
        try {
          const indexList = await this.pinecone.listIndexes();
          console.log('📋 Available indexes (raw):', indexList);
          
          // ✅ FIXED: Properly extract index names from the response
          let indexNames: string[] = [];
          
          if (Array.isArray(indexList)) {
            // If it's an array of strings
            indexNames = indexList;
          } else if (indexList && typeof indexList === 'object') {
            // If it's an object with indexes property (new Pinecone SDK)
            if (indexList.indexes && Array.isArray(indexList.indexes)) {
              indexNames = indexList.indexes.map((i: any) => i.name);
            }
            // If it's an object with array values
            else if (Array.isArray(indexList)) {
              indexNames = indexList;
            }
            // If it's an array-like object
            else if (typeof indexList === 'object') {
              indexNames = Object.values(indexList).flat().filter(v => typeof v === 'string');
            }
          }
          
          console.log('📋 Extracted index names:', indexNames);
          
          if (indexNames.includes(this.indexName)) {
            console.log(`✅ Index "${this.indexName}" found`);
          } else {
            console.error(`❌ Index "${this.indexName}" not found in Pinecone!`);
            console.error('   Available indexes:', indexNames);
            console.error('   Please run: yarn setup-pinecone');
            console.error('   Or create the index manually with dimension 384');
          }
        } catch (error) {
          console.error('⚠️ Could not verify indexes:', error);
          console.log('   This is not critical - continuing with upload/search');
        }
      } else {
        console.warn('⚠️ PINECONE_API_KEY not found - PDF search will be disabled');
      }

      if (!process.env.HUGGINGFACE_API_KEY) {
        console.warn('⚠️ HUGGINGFACE_API_KEY not configured - PDF embeddings will be disabled');
        console.warn('   Please set a valid Hugging Face API key in your .env file');
      } else {
        console.log('✅ Hugging Face API key found for embeddings');
      }
    } catch (error) {
      console.error('Failed to initialize Pinecone service:', error);
      
      // Log specific error details for debugging
      if (error instanceof Error && 'code' in error && error.code === 'ENOENT') {
        console.error('🔍 File not found error during initialization:');
        console.error('   Path:', (error as any).path);
        console.error('   This might be a configuration or dependency issue.');
      }
    }
  }

  async uploadPDF(file: File): Promise<PDFDocument | null> {
    try {
      console.log('📄 Starting PDF upload:', file.name);
      
      if (!this.pinecone) {
        console.error('❌ Pinecone not initialized');
        throw new Error('Pinecone not initialized');
      }

      if (!process.env.HUGGINGFACE_API_KEY) {
        console.error('❌ Hugging Face API key not configured');
        throw new Error('Hugging Face API key not configured');
      }

      // Validate file type and size
      if (!file.name.toLowerCase().endsWith('.pdf')) {
        throw new Error('Only PDF files are supported');
      }
      
      if (file.size === 0) {
        throw new Error('File is empty');
      }
      
      if (file.size > 10 * 1024 * 1024) { // 10MB limit
        throw new Error('File is too large (max 10MB)');
      }

      console.log('📖 Parsing PDF content...');
      // Parse PDF from the actual uploaded file buffer
      const arrayBuffer = await file.arrayBuffer();
      
      if (!arrayBuffer || arrayBuffer.byteLength === 0) {
        throw new Error('Invalid file buffer');
      }
      
      const buffer = Buffer.from(arrayBuffer);
      
      // Pass the buffer directly to pdf-parse
      const pdfData = await safePdfParse(buffer);
      
      if (!pdfData.text || pdfData.text.trim().length === 0) {
        throw new Error('No text content found in PDF');
      }
      
      console.log(`📊 PDF parsed: ${pdfData.numpages} pages, ${pdfData.text.length} characters`);

      // Split into chunks with page tracking (reduced size for API limits)
      const chunksWithPages = this.splitIntoChunksWithPages(pdfData, 500); // 500 words per chunk
      console.log(`🔗 Split into ${chunksWithPages.length} chunks with page tracking`);
      
      // Filter out chunks that are too short or mostly whitespace
      const validChunks = chunksWithPages.filter(chunk => {
        const cleanContent = chunk.content.trim().replace(/\s+/g, ' ');
        const wordCount = cleanContent.split(' ').length;
        const hasRealContent = /[a-zA-Z0-9]/.test(cleanContent);
        
        return cleanContent.length > 20 && wordCount > 5 && hasRealContent;
      });
      
      console.log(`✅ Filtered to ${validChunks.length} valid chunks (removed ${chunksWithPages.length - validChunks.length} empty/short chunks)`);
      
      // Generate embeddings and store
      const index = this.pinecone.index(this.indexName);
      const vectors = [];

      console.log('🧮 Generating Hugging Face embeddings...');
      let successfulEmbeddings = 0;
      let failedEmbeddings = 0;
      
      for (let i = 0; i < validChunks.length; i++) {
        const chunkData = validChunks[i];
        
        try {
          // Clean and validate chunk content
          const cleanContent = chunkData.content.trim().replace(/\s+/g, ' ');
          const wordCount = cleanContent.split(' ').length;
          const charCount = cleanContent.length;
          
          console.log(`   Processing chunk ${i + 1}/${validChunks.length} (${wordCount} words, ${charCount} chars)`);
          
          // ✅ Using Hugging Face API for embeddings (NO GEMINI)
          const embedding = await this.generateEmbedding(cleanContent);
          
          vectors.push({
            id: `${file.name}-chunk-${i}`,
            values: embedding,
            metadata: {
              content: cleanContent,
              filename: file.name,
              chunkIndex: i,
              title: file.name.replace('.pdf', ''),
              uploadDate: new Date().toISOString(),
              page: chunkData.page,
              startPage: chunkData.startPage,
              endPage: chunkData.endPage,
              wordCount: wordCount,
              charCount: charCount,
            }
          });
          successfulEmbeddings++;
          
          // Add delay to avoid rate limiting
          if (i > 0 && i % 5 === 0) {
            console.log(`   Processed ${i} chunks, pausing to avoid rate limits...`);
            await new Promise(resolve => setTimeout(resolve, 500));
          }
          
        } catch (embeddingError) {
          console.error(`❌ Failed to generate embedding for chunk ${i}:`, embeddingError);
          failedEmbeddings++;
        }
      }
      
      console.log(`📊 Embedding Results: ${successfulEmbeddings} successful, ${failedEmbeddings} failed`);
      
      if (vectors.length === 0) {
        throw new Error(`No embeddings were generated successfully. This might be due to API issues.`);
      }

      console.log('💾 Storing vectors in Pinecone...');
      // Use smaller batch size (50) to avoid 502 errors
      const batchSize = 50;
      for (let i = 0; i < vectors.length; i += batchSize) {
        const batch = vectors.slice(i, i + batchSize);
        await index.upsert(batch);
        console.log(`✅ Uploaded batch ${Math.floor(i / batchSize) + 1}/${Math.ceil(vectors.length / batchSize)}`);
      }
      
      console.log('✅ PDF upload completed successfully');

      return {
        id: file.name,
        title: file.name.replace('.pdf', ''),
        content: pdfData.text,
        chunks: chunksWithPages.map(c => c.content),
        metadata: {
          filename: file.name,
          uploadDate: new Date().toISOString(),
          pageCount: pdfData.numpages || 0,
          contentType: 'text' as const,
        }
      };
    } catch (error) {
      console.error('❌ Error uploading PDF:', error);
      
      // Enhanced error handling for common issues
      if (error instanceof Error) {
        if (error.message.includes('PineconeNotFoundError') || error.message.includes('404')) {
          console.error('🔍 Pinecone Index Not Found:');
          console.error(`   The index "${this.indexName}" does not exist.`);
          console.error('   Please run: yarn setup-pinecone');
          console.error('   This will create the required index with correct specifications.');
        } else if (error.message.includes('401') || error.message.includes('authentication')) {
          console.error('🔑 Authentication Error:');
          console.error('   Please check your PINECONE_API_KEY in .env.local');
        } else if (error.message.includes('quota') || error.message.includes('limit')) {
          console.error('📊 Quota/Limit Error:');
          console.error('   You may have reached your Pinecone plan limits.');
          console.error('   Check your usage at: https://app.pinecone.io/');
        }
      }
      
      return null;
    }
  }

  // ✅ ALWAYS run RAG search - No knowledge check, always search PDFs
  async searchSimilarContent(query: string, topK: number = 5): Promise<SearchResult[]> {
    try {
      console.log('🔍 ALWAYS RUNNING RAG SEARCH - Searching for similar content:', query);
      
      if (!this.pinecone) {
        console.warn('⚠️ Pinecone not initialized');
        return [];
      }

      if (!process.env.HUGGINGFACE_API_KEY) {
        console.warn('⚠️ Hugging Face API key not configured');
        return [];
      }

      console.log('🧮 Generating query embedding using Hugging Face (NO GEMINI)...');
      // ✅ Using Hugging Face API for query embedding (NO GEMINI)
      const queryEmbedding = await this.generateEmbedding(query);
      
      console.log(`🔎 Querying Pinecone index: ${this.indexName}...`);
      const index = this.pinecone.index(this.indexName);

      const searchResponse = await index.query({
        vector: queryEmbedding,
        topK,
        includeMetadata: true,
      });

      console.log('📊 Pinecone search response:');
      console.log('   Matches found:', searchResponse.matches?.length || 0);
      
      if (searchResponse.matches && searchResponse.matches.length > 0) {
        console.log('📄 Search results details:');
        searchResponse.matches.forEach((match, idx) => {
          console.log(`   ${idx + 1}. Score: ${match.score?.toFixed(3)}, ID: ${match.id}`);
          console.log(`      Filename: ${match.metadata?.filename}`);
          console.log(`      Page: ${match.metadata?.page}`);
          console.log(`      Content preview: ${(match.metadata?.content as string)?.substring(0, 100)}...`);
        });
        
        console.log(`✅ RAG SUCCESSFUL - Found ${searchResponse.matches.length} relevant PDF chunks`);
      } else {
        console.warn('⚠️ No matches found in Pinecone search');
        console.log('   This could mean:');
        console.log('   - No PDFs have been uploaded yet');
        console.log('   - The query doesn\'t match any content in your PDFs');
        console.log('   - Try uploading a PDF first');
        
        // Try to get some stats about the index
        try {
          const stats = await index.describeIndexStats();
          console.log('📈 Index stats:', stats);
        } catch (statsError) {
          console.log('   Could not retrieve index stats');
        }
      }

      const results = searchResponse.matches?.map(match => ({
        content: match.metadata?.content as string || '',
        score: match.score || 0,
        metadata: match.metadata,
        source: `PDF: ${match.metadata?.filename || 'Unknown'}`,
        page: typeof match.metadata?.page === 'number' ? match.metadata.page : 1,
        startPage: typeof match.metadata?.startPage === 'number' ? match.metadata.startPage : 1,
        endPage: typeof match.metadata?.endPage === 'number' ? match.metadata.endPage : 1
      })) || [];
      
      console.log(`✅ Search completed: ${results.length} results found`);
      return results;
      
    } catch (error) {
      console.error('❌ Error searching Pinecone:', error);
      
      // Enhanced error handling for search operations
      if (error instanceof Error) {
        if (error.message.includes('PineconeNotFoundError') || error.message.includes('404')) {
          console.error('🔍 Pinecone Index Not Found:');
          console.error(`   The index "${this.indexName}" does not exist for search.`);
          console.error('   Please run: yarn setup-pinecone');
        } else if (error.message.includes('401') || error.message.includes('authentication')) {
          console.error('🔑 Authentication Error:');
          console.error('   Please check your PINECONE_API_KEY in .env.local');
        }
      }
      
      return [];
    }
  }

  // ✅ ONLY Hugging Face API - NO GEMINI embeddings anywhere
  private async generateEmbedding(text: string): Promise<number[]> {
    try {
      // Clean and prepare text
      const cleanedText = text.trim().replace(/\s+/g, ' ');
      const charCount = cleanedText.length;
      const wordCount = cleanedText.split(' ').length;
      
      console.log(`🧮 Generating embedding using Hugging Face BAAI/bge-small-en-v1.5...`);
      console.log(`   Text stats: ${wordCount} words, ${charCount} characters`);
      
      // Truncate if too long (BGE model has token limits)
      const maxChars = 5000;
      let processedText = cleanedText;
      
      if (charCount > maxChars) {
        console.warn(`⚠️ Text exceeds limit (${charCount} chars), truncating to ${maxChars}...`);
        processedText = cleanedText.substring(0, maxChars);
        // Ensure we end on a complete word
        const lastSpaceIndex = processedText.lastIndexOf(' ');
        if (lastSpaceIndex > maxChars * 0.8) {
          processedText = processedText.substring(0, lastSpaceIndex);
        }
      }
      
      // ✅ Hugging Face Inference API - NO GEMINI
      const response = await fetch(
        "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5",
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            inputs: processedText,
          }),
        }
      );

      // Handle non-JSON safely
      const textResponse = await response.text();
      
      let result;
      try {
        result = JSON.parse(textResponse);
      } catch (err) {
        console.error(`Raw HF response:`, textResponse);
        throw new Error('Invalid JSON from HuggingFace');
      }

      console.log(`HF Response received:`, Array.isArray(result) ? `${result.length} items` : typeof result);

      // ✅ Flexible embedding extraction
      let embedding;
      
      if (Array.isArray(result) && typeof result[0] === "number") {
        // Case 1: Direct 1D embedding
        embedding = result;
      } else if (Array.isArray(result) && Array.isArray(result[0])) {
        // Case 2: 2D embedding
        embedding = result[0];
      } else if (result?.embeddings) {
        embedding = result.embeddings;
      } else {
        console.error("Invalid embedding format:", result);
        throw new Error('Invalid embedding format from HuggingFace');
      }
      
      if (!embedding || !Array.isArray(embedding) || embedding.length === 0) {
        throw new Error('Empty embedding received');
      }

      console.log(`✅ Generated embedding with ${embedding.length} dimensions`);
      return embedding;
      
    } catch (error) {
      console.error('❌ Error generating Hugging Face embedding:', error);
      
      // Enhanced error handling
      if (error instanceof Error) {
        if (error.message.includes('API_KEY') || error.message.includes('Authorization')) {
          console.error('🔑 Invalid Hugging Face API Key:');
          console.error('   Your HUGGINGFACE_API_KEY in .env.local is not valid.');
          console.error('   Please get a valid API key from: https://huggingface.co/settings/tokens');
        } else if (error.message.includes('rate') || error.message.includes('limit') || error.message.includes('429')) {
          console.error('📊 Hugging Face API Rate Limit Error:');
          console.error('   You may have exceeded your API usage limits.');
          console.error('   Consider adding longer delays between requests.');
        }
      }
      
      throw error;
    }
  }

  private splitIntoChunksWithPages(pdfData: any, chunkSize: number): Array<{content: string, page: number, startPage: number, endPage: number}> {
    const text = pdfData.text;
    
    // Clean the text first - remove excessive whitespace and newlines
    const cleanedText = text
      .replace(/\n{3,}/g, '\n\n') // Replace 3+ newlines with 2
      .replace(/\s{3,}/g, ' ')     // Replace 3+ spaces with 1
      .trim();
    
    console.log(`📝 Text cleaning: ${text.length} -> ${cleanedText.length} characters`);
    
    // Split by sentences first, then by words if needed
    const sentences = cleanedText.split(/[.!?]+/).filter((s: string) => s.trim().length > 0);
    const chunks: Array<{content: string, page: number, startPage: number, endPage: number}> = [];
    
    // Estimate pages based on character count (approximate)
    const avgCharsPerPage = Math.ceil(cleanedText.length / (pdfData.numpages || 1));
    
    let currentChunk = '';
    let currentWordCount = 0;
    let chunkStartPos = 0;
    
    for (const sentence of sentences) {
      const sentenceWords = sentence.trim().split(' ').length;
      const sentenceText = sentence.trim() + '. ';
      
      // If adding this sentence would exceed chunk size, finalize current chunk
      if (currentWordCount + sentenceWords > chunkSize && currentChunk.length > 0) {
        // Calculate page information for current chunk
        const chunkEnd = chunkStartPos + currentChunk.length;
        const startPage = Math.max(1, Math.ceil(chunkStartPos / avgCharsPerPage));
        const endPage = Math.min(pdfData.numpages || 1, Math.ceil(chunkEnd / avgCharsPerPage));
        const page = startPage;
        
        chunks.push({
          content: currentChunk.trim(),
          page,
          startPage,
          endPage
        });
        
        // Start new chunk
        chunkStartPos = chunkEnd;
        currentChunk = sentenceText;
        currentWordCount = sentenceWords;
      } else {
        // Add sentence to current chunk
        currentChunk += sentenceText;
        currentWordCount += sentenceWords;
      }
    }
    
    // Add final chunk if it has content
    if (currentChunk.trim().length > 0) {
      const chunkEnd = chunkStartPos + currentChunk.length;
      const startPage = Math.max(1, Math.ceil(chunkStartPos / avgCharsPerPage));
      const endPage = Math.min(pdfData.numpages || 1, Math.ceil(chunkEnd / avgCharsPerPage));
      const page = startPage;
      
      chunks.push({
        content: currentChunk.trim(),
        page,
        startPage,
        endPage
      });
    }
    
    console.log(`🔗 Created ${chunks.length} sentence-based chunks`);
    return chunks;
  }

  async deleteDocument(filename: string): Promise<boolean> {
    try {
      if (!this.pinecone) {
        return false;
      }

      const index = this.pinecone.index(this.indexName);
      
      // Get embedding dimension from Hugging Face model (384 for BAAI/bge-small-en-v1.5)
      const dimension = 384;
      
      // Find all chunks for this document
      const searchResponse = await index.query({
        vector: new Array(dimension).fill(0),
        topK: 10000,
        includeMetadata: true,
        filter: { filename: { $eq: filename } }
      });

      if (searchResponse.matches && searchResponse.matches.length > 0) {
        const idsToDelete = searchResponse.matches.map(match => match.id);
        // Delete in batches
        const batchSize = 100;
        for (let i = 0; i < idsToDelete.length; i += batchSize) {
          const batch = idsToDelete.slice(i, i + batchSize);
          await index.deleteMany(batch);
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
}

export const pineconeService = new PineconeService();