# Location Review Finder

A Next.js application that allows users to search for locations and find reviews using Mapbox Geocoding API and Google Places API.

## Features

- **Dynamic Search**: Real-time location search with 1-second debouncing
- **Mapbox Geocoding**: Fast and accurate location suggestions using Mapbox API
- **Dropdown Results**: Interactive dropdown showing search suggestions with keyboard navigation
- **Place Details**: Detailed information and reviews from Google Places API
- **Responsive Design**: Mobile-friendly interface built with Tailwind CSS

## Getting Started

### Prerequisites

You'll need API keys for:
1. **Mapbox**: Get a free access token from [Mapbox](https://account.mapbox.com/)
2. **Google Places API**: Get an API key from [Google Cloud Console](https://console.cloud.google.com/)

### Environment Setup

1. Copy the environment template:
```bash
cp .env.example .env.local
```

2. Add your API keys to `.env.local`:
```env
MAPBOX_ACCESS_TOKEN=your_mapbox_access_token_here
GOOGLE_PLACES_API_KEY=your_google_places_api_key_here
```

### Installation

```bash
npm install
```

### Development

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

### Building

```bash
npm run build
npm start
```

## How It Works

1. **Search Input**: User types a location query (business name, address, etc.)
2. **Debounced Search**: After 1 second of inactivity, the app calls Mapbox Geocoding API
3. **Dropdown Results**: Search results appear in a dropdown with keyboard navigation support
4. **Location Selection**: User selects a location from dropdown or presses Enter
5. **Place Details**: App fetches detailed information and reviews from Google Places API

## API Endpoints

- `GET /api/geocoding?q=query` - Search locations using Mapbox Geocoding
- `POST /api/places/search` - Get place details and reviews from Google Places

## Tech Stack

- **Next.js 15** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Mapbox Geocoding API** - Location search
- **Google Places API** - Place details and reviews

## Project Structure

```
frontend/
├── app/
│   ├── api/
│   │   ├── geocoding/route.ts    # Mapbox geocoding endpoint
│   │   └── places/search/route.ts # Google Places endpoint
│   ├── page.tsx                  # Main application
│   └── layout.tsx               # App layout
├── components/ui/               # UI components
├── hooks/                       # Custom React hooks
└── lib/                        # Utility functions
```
