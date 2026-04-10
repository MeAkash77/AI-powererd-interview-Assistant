// ✅ REMOVED Gemini completely - using simple fallback logic
// No more GoogleGenerativeAI import or usage

export interface ExtractedQuestion {
  question: string;
  context: string;
  confidence: number;
}

/**
 * Extract question from transcript using simple fallback logic
 * No API calls - completely local and reliable
 */
export function extractQuestionFromTranscript(transcript: string): ExtractedQuestion | null {
  try {
    console.log('🔍 Extracting question using simple fallback logic...');
    
    if (!transcript || transcript.trim().length === 0) {
      console.log('❌ Transcript is empty');
      return null;
    }

    // Clean the transcript
    const cleanedTranscript = transcript.trim();
    console.log(`📝 Transcript length: ${cleanedTranscript.length} characters`);

    // Split into sentences
    const sentences = cleanedTranscript.split(/[.?!]+/);
    const nonEmptySentences = sentences.filter(s => s.trim().length > 0);
    
    console.log(`📊 Found ${nonEmptySentences.length} sentences`);

    if (nonEmptySentences.length === 0) {
      console.log('❌ No sentences found in transcript');
      return null;
    }

    // Get the last sentence (most recent question/statement)
    const lastSentence = nonEmptySentences[nonEmptySentences.length - 1].trim();
    console.log(`📝 Last sentence: "${lastSentence.substring(0, 100)}..."`);

    // Check if the last sentence looks like a question
    const isQuestion = /[?]$/.test(lastSentence) || 
                      /^(what|why|how|when|where|who|can|could|would|should|is|are|do|does|did)/i.test(lastSentence);
    
    let question = lastSentence;
    let confidence = 0.5;
    
    if (isQuestion) {
      confidence = 0.8;
      console.log('✅ Detected question pattern in last sentence');
    } else {
      // If last sentence isn't a question, try to find the last question in the transcript
      const questions = nonEmptySentences.filter(s => /[?]$/.test(s) || 
                                                      /^(what|why|how|when|where|who|can|could|would|should)/i.test(s));
      
      if (questions.length > 0) {
        question = questions[questions.length - 1].trim();
        confidence = 0.7;
        console.log(`✅ Found question in transcript: "${question.substring(0, 100)}..."`);
      } else {
        console.log('⚠️ No clear question pattern found, using last sentence as fallback');
      }
    }

    // Get context around the question (previous 2 sentences)
    const sentenceIndex = nonEmptySentences.findIndex(s => s.trim() === question);
    let context = '';
    
    if (sentenceIndex !== -1) {
      const startIdx = Math.max(0, sentenceIndex - 2);
      const endIdx = Math.min(nonEmptySentences.length - 1, sentenceIndex + 1);
      context = nonEmptySentences.slice(startIdx, endIdx + 1).join('. ') + '.';
    } else {
      // Fallback: use first 200 characters as context
      context = cleanedTranscript.substring(0, 200);
    }

    // Don't accept questions that are too short
    if (question.length < 5) {
      console.log('❌ Question too short, skipping');
      return null;
    }

    // Don't accept generic phrases
    const genericPhrases = ['hello', 'hi', 'yes', 'no', 'ok', 'okay', 'thanks', 'thank you'];
    if (genericPhrases.some(phrase => question.toLowerCase().includes(phrase)) && question.length < 20) {
      console.log('❌ Generic phrase detected, skipping');
      return null;
    }

    console.log(`✅ Question extracted successfully (confidence: ${confidence})`);
    console.log(`   Question: ${question.substring(0, 150)}...`);
    
    return {
      question: question,
      context: context.substring(0, 500),
      confidence: confidence
    };
    
  } catch (error) {
    console.error('❌ Error in simple question extraction:', error);
    return null;
  }
}

/**
 * Alternative: Extract last sentence as question (simplest approach)
 * Use this if the above method doesn't work well
 */
export function extractLastSentenceAsQuestion(transcript: string): string | null {
  try {
    if (!transcript || transcript.trim().length === 0) {
      return null;
    }
    
    const sentences = transcript.split(/[.?!]+/);
    const nonEmptySentences = sentences.filter(s => s.trim().length > 0);
    
    if (nonEmptySentences.length === 0) {
      return null;
    }
    
    return nonEmptySentences[nonEmptySentences.length - 1].trim();
  } catch (error) {
    console.error('Error extracting last sentence:', error);
    return null;
  }
}

/**
 * Alternative: Find the last question mark in transcript
 */
export function extractLastQuestion(transcript: string): string | null {
  try {
    // Find the last question mark
    const lastQuestionIndex = transcript.lastIndexOf('?');
    
    if (lastQuestionIndex === -1) {
      // If no question mark, try to find question pattern
      const questionPattern = /(?:what|why|how|when|where|who|can|could|would|should)[^.!?]+/gi;
      const matches = transcript.match(questionPattern);
      
      if (matches && matches.length > 0) {
        return matches[matches.length - 1].trim();
      }
      return null;
    }
    
    // Find the start of the question (look back for sentence start)
    let startIndex = lastQuestionIndex;
    for (let i = lastQuestionIndex; i >= 0; i--) {
      if (i === 0 || transcript[i - 1] === '.' || transcript[i - 1] === '!' || transcript[i - 1] === '?') {
        startIndex = i;
        break;
      }
    }
    
    const question = transcript.substring(startIndex, lastQuestionIndex + 1).trim();
    return question || null;
  } catch (error) {
    console.error('Error extracting last question:', error);
    return null;
  }
}