#!/usr/bin/env python3
"""
📥 LottoGenius - Daten-Abruf-System
Holt aktuelle Lotto-Zahlen von der API
"""
import requests
import json
from datetime import datetime
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
API_URL = "https://johannesfriedrich.github.io/LottoNumberArchive/Lottonumbers_tidy_complete.json"

def fetch_lotto_data():
    print("=" * 60)
    print("📥 LottoGenius - Daten-Abruf")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()
    
    print(f"🌐 Lade Daten von API...")
    
    try:
        response = requests.get(API_URL, timeout=60)
        response.raise_for_status()
        raw_data = response.json()
        
        print(f"   ✅ {len(raw_data)} Datensätze empfangen")
        
        # Verarbeite Daten
        draws = {}
        for entry in raw_data:
            date = entry.get('date', '')
            if not date:
                continue
            
            if date not in draws:
                draws[date] = {'date': date, 'numbers': [], 'superzahl': None}
            
            variable = entry.get('variable', '')
            value = entry.get('value')
            
            if variable.startswith('Lottozahl'):
                draws[date]['numbers'].append(value)
            elif variable == 'Superzahl':
                draws[date]['superzahl'] = value
        
        # Filtere vollständige Ziehungen
        complete_draws = []
        for date, draw in draws.items():
            if len(draw['numbers']) == 6 and draw['superzahl'] is not None:
                draw['numbers'].sort()
                complete_draws.append(draw)
        
        # Sortiere nach Datum (neueste zuerst)
        complete_draws.sort(
            key=lambda x: datetime.strptime(x['date'], '%d.%m.%Y'), 
            reverse=True
        )
        
        # Speichere
        os.makedirs(DATA_DIR, exist_ok=True)
        
        output = {
            'last_update': datetime.now().isoformat(),
            'total_draws': len(complete_draws),
            'draws': complete_draws
        }
        
        with open(os.path.join(DATA_DIR, 'lotto_data.json'), 'w') as f:
            json.dump(output, f, indent=2)
        
        print()
        print(f"✅ {len(complete_draws)} vollständige Ziehungen gespeichert")
        print()
        print(f"📅 Letzte Ziehung: {complete_draws[0]['date']}")
        print(f"🎱 Zahlen: {complete_draws[0]['numbers']}")
        print(f"⭐ Superzahl: {complete_draws[0]['superzahl']}")
        print()
        print("=" * 60)
        
        return complete_draws
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return []

if __name__ == "__main__":
    fetch_lotto_data()
