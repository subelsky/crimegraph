#!/usr/bin/env ruby

require 'json'
require 'csv'

experiment_id, hyperparameter = ARGV
hyperparameter = hyperparameter.to_sym

puts "Reading results for experiment: #{experiment_id}, hyperparameter: #{hyperparameter}"

base_dir = "#{Dir.pwd}/notebooks/model_training_and_tuning/tuning/trial_#{experiment_id}"
subdirectories = Dir.glob("#{base_dir}/**/**/trial.json").map! { |file| File.dirname(file) }

headers = %i(trial_id)
headers << hyperparameter
headers.concat(%i[loss val_loss hit_rate val_hit_rate])

output_path = "#{Dir.pwd}/data/tuning_results/experiment_#{experiment_id}.csv"

CSV.open(output_path, 'w', headers:, write_headers: true) do |csv|
  subdirectories.each do |subdirectory|
    # puts subdirectory
    data = JSON.load_file!(File.join(subdirectory, 'trial.json'), symbolize_names: true)

    data => {
      trial_id:,
      hyperparameters: { values: },
      metrics: { metrics: },
      status:
    }

    next if status == 'RUNNING'

    hyperparameter_value = values.fetch(hyperparameter)

    metrics = metrics.inject({}) do |total, (k,v)|
      total.merge!(k => v[:observations][0].fetch(:value)[0])
    end

    output = { trial_id:, hyperparameter => hyperparameter_value }.merge(metrics)
    csv << output
  end
end

puts "Wrote results to: #{output_path}"
