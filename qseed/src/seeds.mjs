// Curated real head search terms per vertical. These seed the semi-synthetic
// keyword bank (each is expanded into long-tail variants in data.mjs), mirroring
// how a commercial keyword tool grows a seed list. Terms chosen from widely
// searched, real head keywords in each domain.
export const SEEDS = {
  consumer_electronics: [
    'noise cancelling headphones', 'wireless earbuds', '4k monitor', 'gaming laptop',
    'mechanical keyboard', 'smartphone', 'smartwatch', 'robot vacuum', 'air fryer',
    'soundbar', 'oled tv', 'tablet', 'e-reader', 'portable ssd', 'webcam', 'dash cam',
    'home security camera', 'bluetooth speaker', 'gaming mouse', 'graphics card',
    'wifi router', 'power bank', 'standing desk', 'ergonomic office chair', 'drone',
    'action camera', 'projector', 'electric toothbrush', 'espresso machine', 'vr headset',
    'fitness tracker', 'usb-c hub', 'external hard drive', 'wireless charger',
    'gaming monitor', 'budget smartphone', 'laptop for students', 'streaming device',
    'wireless printer', 'noise cancelling earbuds',
  ],
  personal_finance: [
    'high yield savings account', 'best credit card', 'roth ira', 'index funds',
    'refinance mortgage', 'personal loan', 'improve credit score', 'debt consolidation',
    'budgeting app', 'tax software', 'life insurance', 'car insurance',
    'student loan refinance', 'brokerage account', '401k rollover', 'emergency fund',
    'cryptocurrency investing', 'robo advisor', 'balance transfer card', 'home equity loan',
    'first time home buyer', 'retirement calculator', 'stock trading app', 'dividend stocks',
    'health savings account', 'financial advisor', 'current mortgage rates', 'cash back card',
    'travel rewards card', 'no fee checking account', 'pay off debt fast', 'invest 1000 dollars',
    'side hustle income', 'inflation hedge', 'bonds vs stocks', 'credit repair',
    'track net worth', 'passive income ideas', 'money market account', 'certificate of deposit',
  ],
  health_wellness: [
    'weight loss diet', 'intermittent fasting', 'best protein powder', 'vitamin d supplement',
    'probiotics', 'meditation for anxiety', 'how to sleep better', 'lower blood pressure',
    'keto diet', 'home workout', 'running for beginners', 'magnesium supplement', 'collagen',
    'omega 3', 'boost immune system', 'gut health', 'back pain relief', 'hydration',
    'healthy meal prep', 'reduce stress', 'blood sugar control', 'cholesterol lowering foods',
    'yoga for beginners', 'strength training', 'creatine', 'mental health apps',
    'seasonal allergies', 'migraine relief', 'quit smoking', 'anti inflammatory diet',
    'multivitamin', 'sleep supplement', 'cardio vs weights', 'calorie deficit',
    'fiber supplement', 'joint pain relief', 'natural energy boost', 'skincare routine',
    'mediterranean diet', 'lower resting heart rate',
  ],
};

export const VERTICAL_LABEL = {
  consumer_electronics: 'consumer electronics and gadgets',
  personal_finance: 'personal finance, banking, credit, and investing',
  health_wellness: 'health, fitness, nutrition, and wellness',
};
