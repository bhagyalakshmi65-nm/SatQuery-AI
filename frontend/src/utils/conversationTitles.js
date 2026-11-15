/**
 * Conversation Title and Context Utilities for SatQuery AI
 * Provides semantic title generation and base64-to-File context restoration.
 */

/**
 * Generates a concise, human-readable title (2-4 words) for a conversation
 * based on the initial user inquiry.
 *
 * @param {string} queryText - The initial user prompt
 * @returns {string} Short, elegant conversation title
 */
export function generateConversationTitle(queryText = '') {
  const text = (queryText || '').toLowerCase().trim();
  if (!text) return 'Satellite image analysis';

  // 0. Location + Temporal queries (e.g. Bengaluru 2022-2026)
  const cities = ['bengaluru', 'bangalore', 'hyderabad', 'new delhi', 'delhi', 'mumbai', 'chennai', 'pune', 'kolkata'];
  const foundCity = cities.find((c) => text.includes(c));
  const yearMatch = text.match(/\b(20\d\d)\b/g);
  if (foundCity && yearMatch && yearMatch.length >= 2) {
    const cityName = foundCity === 'bangalore' ? 'Bengaluru' : foundCity.split(' ').map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    return `${cityName} (${yearMatch[0]}–${yearMatch[1]})`;
  }
  if (foundCity) {
    const cityName = foundCity === 'bangalore' ? 'Bengaluru' : foundCity.split(' ').map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    return `${cityName} Earth Observation`;
  }

  // 1. Comparison / Change detection
  if (
    text.includes('change') ||
    text.includes('compare') ||
    text.includes('difference') ||
    text.includes('before and after') ||
    text.includes('temporal')
  ) {
    return 'Satellite image comparison';
  }

  // 2. Building / Structure detection
  if (
    text.includes('building') ||
    text.includes('house') ||
    text.includes('structure') ||
    text.includes('rooftop')
  ) {
    return 'Building detection';
  }

  // 3. Vegetation / Forest / Agriculture
  if (
    text.includes('vegetation') ||
    text.includes('forest') ||
    text.includes('tree') ||
    text.includes('crop') ||
    text.includes('agriculture') ||
    text.includes('greenery')
  ) {
    return 'Vegetation analysis';
  }

  // 4. Water / Flooding / Ocean / River / Coastal
  if (
    text.includes('flood') ||
    text.includes('water') ||
    text.includes('river') ||
    text.includes('lake') ||
    text.includes('ocean') ||
    text.includes('coastal')
  ) {
    if (text.includes('flood')) return 'Flood analysis';
    return 'Water extent analysis';
  }

  // 5. Land Cover / Land Use
  if (text.includes('land cover') || text.includes('landcover') || text.includes('land use')) {
    return 'Land cover classification';
  }

  // 6. Urban / Infrastructure / Roads
  if (text.includes('urban') || text.includes('rural') || text.includes('city')) {
    return 'Urban area analysis';
  }
  if (text.includes('road') || text.includes('highway') || text.includes('infrastructure')) {
    return 'Infrastructure mapping';
  }

  // 7. Vessels / Aircraft
  if (text.includes('ship') || text.includes('vessel') || text.includes('boat') || text.includes('port')) {
    return 'Vessel detection';
  }
  if (text.includes('aircraft') || text.includes('plane') || text.includes('airport')) {
    return 'Airport analysis';
  }

  // 8. General overview / Description
  if (
    text.includes('what is in') ||
    text.includes('what are there') ||
    text.includes('describe') ||
    text.includes('analyze this') ||
    text.includes('tell me about') ||
    text.includes('what do you see') ||
    text.includes('what is this')
  ) {
    return 'Satellite image analysis';
  }

  // 9. Generic clean fallback: strip leading conversational words, take first 3 words
  const cleaned = text
    .replace(/^(can you|please|tell me|how many|how much|what is|what are|where are|is there|are there|tell about|analyze)\s+/i, '')
    .replace(/[^\w\s]/g, '')
    .trim();

  const words = cleaned.split(/\s+/).filter(Boolean).slice(0, 3);
  if (words.length > 0) {
    const capitalized = words.map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    return capitalized.slice(0, 30);
  }

  return 'Satellite image analysis';
}

/**
 * Converts a base64 data URL back into a standard JavaScript File object.
 * Used for restoring imagery context when a user sends follow-up inquiries
 * in a conversation that was reopened from localStorage.
 *
 * @param {string} dataUrl - The base64 data URL
 * @param {string} filename - Fallback filename
 * @returns {Promise<File|null>} Reconstructed File object
 */
export async function dataUrlToFile(dataUrl, filename = 'satellite_scene.jpg') {
  if (!dataUrl || typeof dataUrl !== 'string' || !dataUrl.startsWith('data:')) {
    return null;
  }
  try {
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    return new File([blob], filename, { type: blob.type || 'image/jpeg' });
  } catch (err) {
    console.warn('Unable to convert dataUrl to File:', err);
    return null;
  }
}

