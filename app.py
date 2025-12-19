from flask import Flask, render_template, request, jsonify
import grover_genomics_demo
import blast_search
import traceback
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        data = request.json
        genome = data.get('genome', '')
        motif = data.get('motif', '')
        
        # Validação básica
        if not genome or not motif:
            return jsonify({'error': 'Genoma e Motif são obrigatórios'}), 400
            
        # Executa a lógica clássica para pegar as janelas (para visualização)
        start_classical = time.perf_counter()
        windows, good_indices = grover_genomics_demo.classical_find_occurrences(genome, motif)
        classical_time = time.perf_counter() - start_classical
        
        # Executa a simulação quântica (50 trials)
        # Nota: O script original exige N=potência de 2. Vamos tentar rodar e capturar erro se falhar.
        start_time = time.perf_counter()
        acc, counts, sample_good = grover_genomics_demo.run_trials(genome, motif, trials=50)
        grover_time = time.perf_counter() - start_time
        
        # Prepara dados para o gráfico
        # counts é um Counter {indice: frequencia}
        # Vamos normalizar para porcentagem
        total_shots = sum(counts.values())
        results = []
        for i in range(len(windows)):
            count = counts.get(i, 0)
            probability = (count / total_shots) * 100
            is_target = i in good_indices
            results.append({
                'index': i,
                'window_content': windows[i],
                'probability': probability,
                'is_target': is_target,
                'count': count
            })

        histogram = [
            {
                'index': idx,
                'count': count,
                'probability': (count / total_shots) * 100,
                'is_target': idx in good_indices
            }
            for idx, count in sorted(counts.items())
        ]
            
        return jsonify({
            'accuracy': acc,
            'results': results,
            'total_windows': len(windows),
            'good_indices': good_indices,
            'execution_time': grover_time,
            'classical_time': classical_time,
            'total_shots': total_shots,
            'histogram': histogram
        })

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f"Erro interno: {str(e)}"}), 500


@app.route('/blast', methods=['POST'])
def blast():
    """Execute BLAST-like search only."""
    try:
        data = request.json
        genome = data.get('genome', '')
        motif = data.get('motif', '')
        threshold = data.get('threshold', 70.0)
        
        if not genome or not motif:
            return jsonify({'error': 'Genoma e Motif são obrigatórios'}), 400
        
        result = blast_search.blast_search(genome, motif, threshold)
        return jsonify(result)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f"Erro interno: {str(e)}"}), 500


@app.route('/compare', methods=['POST'])
def compare():
    """
    Compare Grover quantum search vs BLAST-like search.
    Returns both results side by side with analysis.
    """
    try:
        data = request.json
        genome = data.get('genome', '')
        motif = data.get('motif', '')
        blast_threshold = data.get('blast_threshold', 70.0)
        
        if not genome or not motif:
            return jsonify({'error': 'Genoma e Motif são obrigatórios'}), 400
        
        # Execute Grover
        start_classical = time.perf_counter()
        windows, good_indices = grover_genomics_demo.classical_find_occurrences(genome, motif)
        classical_time = time.perf_counter() - start_classical
        
        start_grover = time.perf_counter()
        acc, counts, _ = grover_genomics_demo.run_trials(genome, motif, trials=50)
        grover_time = time.perf_counter() - start_grover
        
        total_shots = sum(counts.values())
        histogram = [
            {
                'index': idx,
                'count': count,
                'probability': (count / total_shots) * 100,
                'is_target': idx in good_indices
            }
            for idx, count in sorted(counts.items())
        ]
        grover_results = {
            'accuracy': acc,
            'good_indices': good_indices,
            'total_windows': len(windows),
            'execution_time': grover_time,
            'classical_time': classical_time,
            'total_shots': total_shots,
            'histogram': histogram,
            'results': [
                {
                    'index': i,
                    'window_content': windows[i],
                    'probability': (counts.get(i, 0) / total_shots) * 100,
                    'is_target': i in good_indices,
                    'count': counts.get(i, 0)
                }
                for i in range(len(windows))
            ]
        }
        
        # Execute comparison
        comparison = blast_search.compare_methods(
            genome, motif, grover_results, blast_threshold
        )
        
        # Add Grover detailed results
        comparison['grover']['execution_time'] = grover_time
        comparison['grover']['classical_time'] = classical_time
        comparison['grover']['results'] = grover_results['results']
        comparison['grover']['total_windows'] = len(windows)
        
        return jsonify(comparison)
    
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f"Erro interno: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
