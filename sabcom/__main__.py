import sys
import click
import pickle
from runner import runner

@click.group()
@click.version_option("1.0.0")
def main():
    """
    An open source, easy-to-use-and-adapt, spatial network, multi-agent, model
    that can be used to simulate the effects of different lockdown policy measures
    on the spread of the Covid-19 virus in several (South African) cities.
    """
    pass


@main.command()
@click.argument('output_folder_path', required=True)
@click.argument('initialisation_path', required=False)
@click.argument('parameters_path', required=False)
@click.argument('input_folder_path', required=False)
@click.argument('data_output_mode', required=False)
@click.argument('scenario', required=False)
def simulate(output_folder_path, initialisation_path, parameters_path, input_folder_path,
             data_output_mode, scenario):
    """Simulate the model"""
    data = open(initialisation_path, "rb")
    list_of_objects = pickle.load(data)
    environment = list_of_objects[0]

    initial_infections = 


    environment = runner(environment, initial_infections, seed, data_folder=input_folder_path,
                         data_output=output_folder_path)


    click.echo('Simulation done, check out the output data here: {}'.format(output_folder_path))


@main.command()
@click.argument('name', required=False)
def initialise(**kwargs):
    """Initialise the model in specified directory"""
    details = lookup_cve(kwargs.get("name"))
    click.echo(f'CVE-ID \n\n{details["cve-id"]}\n')
    click.echo(f'Description \n\n{details["description"]}\n')
    click.echo(f'References \n\n{details["references"]}\n')
    click.echo(f'Assigning CNA \n\n{details["assigning cna"]}\n')
    click.echo(f'Date Entry \n\n{details["date entry created"]}')


if __name__ == '__main__':
    args = sys.argv
    if "--help" in args or len(args) == 1:
        print("SABCoM")
    main()
