#!/usr/bin/env python3
"""
Real-World Business Case Example

This example demonstrates practical applications of Shapley values in
business scenarios including cost allocation, profit sharing, and
resource optimization.

Scenarios covered:
1. Joint venture profit sharing
2. Shared service cost allocation
3. Sales team commission distribution
"""

from shapley_value import ShapleyValueCalculator, ShapleyCombinations
import json


def joint_venture_example():
    """
    Scenario: Three companies forming a joint venture
    
    Companies contribute different resources and capabilities:
    - TechCorp: Advanced technology (value: 100)
    - MarketInc: Market access and brand (value: 80)
    - LogisCo: Supply chain and logistics (value: 60)
    
    The joint venture creates synergies when companies work together.
    """
    print("=" * 70)
    print("SCENARIO 1: Joint Venture Profit Sharing")
    print("=" * 70)
    
    companies = {
        'TechCorp': 100,    # Technology contribution
        'MarketInc': 80,    # Market access contribution
        'LogisCo': 60       # Logistics contribution
    }
    
    print("Joint Venture Partners:")
    for company, contribution in companies.items():
        print(f"  {company}: ${contribution}M contribution value")
    print()
    
    def joint_venture_value(coalition):
        """
        Calculate the total value created by a coalition of companies.
        
        Synergies:
        - Technology + Market: 50% bonus
        - Technology + Logistics: 30% bonus
        - Market + Logistics: 25% bonus
        - All three together: Additional 20% bonus on top of pair bonuses
        """
        if not coalition:
            return 0
        
        base_value = sum(coalition)
        synergy_bonus = 0
        
        if len(coalition) == 1:
            # Individual companies have some inefficiencies
            return base_value * 0.9
        
        company_names = []
        for value in coalition:
            for name, val in companies.items():
                if val == value:
                    company_names.append(name)
                    break
        
        # Pair synergies
        if 'TechCorp' in company_names and 'MarketInc' in company_names:
            synergy_bonus += base_value * 0.5
        if 'TechCorp' in company_names and 'LogisCo' in company_names:
            synergy_bonus += base_value * 0.3
        if 'MarketInc' in company_names and 'LogisCo' in company_names:
            synergy_bonus += base_value * 0.25
        
        # Triple synergy bonus
        if len(coalition) == 3:
            synergy_bonus += base_value * 0.2
        
        return base_value + synergy_bonus
    
    # Calculate fair profit sharing
    players = list(companies.values())
    calculator = ShapleyValueCalculator(joint_venture_value, players)
    shapley_values = calculator.calculate_shapley_values()
    
    # Display results
    company_names = list(companies.keys())
    total_profit = sum(shapley_values.values())
    
    print("Fair Profit Distribution (Shapley Values):")
    for i, (contribution, profit_share) in enumerate(shapley_values.items()):
        company = company_names[i]
        percentage = (profit_share / total_profit) * 100
        roi = (profit_share / contribution - 1) * 100
        print(f"  {company}:")
        print(f"    - Profit share: ${profit_share:.2f}M ({percentage:.1f}%)")
        print(f"    - ROI: {roi:.1f}%")
    
    print(f"\nTotal venture value: ${total_profit:.2f}M")
    print(f"Value created through collaboration: ${total_profit - sum(companies.values()):.2f}M")


def cost_allocation_example():
    """
    Scenario: Shared IT infrastructure cost allocation
    
    Three departments share a common IT infrastructure. The cost should be
    allocated fairly based on usage patterns and economies of scale.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: Shared Service Cost Allocation")
    print("=" * 70)
    
    departments = {
        'Engineering': 45,  # High compute requirements
        'Marketing': 25,    # Medium requirements
        'Sales': 20,        # Medium requirements
    }
    
    print("Department IT Requirements (standalone cost):")
    for dept, cost in departments.items():
        print(f"  {dept}: ${cost}K")
    print(f"  Total if separate: ${sum(departments.values())}K")
    print()
    
    # Coalition values based on economies of scale (simplified for 3 departments)
    coalition_values = {
        (): 0,
        ('Engineering',): 40,     # Slight economy for single large user
        ('Marketing',): 25,
        ('Sales',): 20,
        ('Engineering', 'Marketing'): 60,     # Good economies of scale
        ('Engineering', 'Sales'): 55,
        ('Marketing', 'Sales'): 40,
        ('Engineering', 'Marketing', 'Sales'): 75,  # Maximum economies
    }
    
    # Calculate fair cost allocation
    players = list(departments.keys())
    calculator = ShapleyCombinations(players)
    shapley_values = calculator.calculate_shapley_values(coalition_values)
    
    print("Fair Cost Allocation:")
    total_allocated = sum(shapley_values.values())
    savings_total = sum(departments.values()) - total_allocated
    
    for dept, allocated_cost in shapley_values.items():
        standalone_cost = departments[dept]
        savings = standalone_cost - allocated_cost
        savings_pct = (savings / standalone_cost) * 100
        allocation_pct = (allocated_cost / total_allocated) * 100
        
        print(f"  {dept}:")
        print(f"    - Allocated cost: ${allocated_cost:.2f}K ({allocation_pct:.1f}%)")
        print(f"    - Savings vs standalone: ${savings:.2f}K ({savings_pct:.1f}%)")
    
    print(f"\nTotal allocated: ${total_allocated:.2f}K")
    print(f"Total savings achieved: ${savings_total:.2f}K")


def sales_team_example():
    """
    Scenario: Sales team commission distribution
    
    A sales team closed a major deal. The commission should be distributed
    fairly based on each member's contribution to the deal.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 3: Sales Team Commission Distribution")
    print("=" * 70)
    
    # Deal worth $1M with 10% commission = $100K to distribute
    total_commission = 100000
    
    sales_team = {
        'Account_Manager': 0.4,    # Lead relationship, deal coordination
        'Technical_Sales': 0.3,    # Technical expertise, demos
        'Sales_Engineer': 0.25,    # Solution design, technical support
        'Sales_Director': 0.15     # Strategy, deal approval, client escalation
    }
    
    print(f"Major deal closed: $1M (${total_commission:,} commission to distribute)")
    print("\nTeam member base contribution rates:")
    for member, rate in sales_team.items():
        print(f"  {member.replace('_', ' ')}: {rate:.0%}")
    print()
    
    def deal_contribution(coalition):
        """
        Calculate the probability of closing the deal with given team members.
        
        Individual contributions are multiplicative (team effort required),
        but there are collaboration bonuses.
        """
        if not coalition:
            return 0
        
        # Convert coalition values back to member names
        member_names = []
        team_list = list(sales_team.keys())
        for rate in coalition:
            for name, member_rate in sales_team.items():
                if abs(member_rate - rate) < 0.001:  # Float comparison
                    member_names.append(name)
                    break
        
        # Base probability is product of individual contributions
        base_prob = 1.0
        for rate in coalition:
            base_prob *= (0.3 + rate)  # Minimum 30% + individual contribution
        
        # Collaboration bonuses
        collaboration_bonus = 0
        
        # Account Manager + Technical Sales: Strong combo
        if 'Account_Manager' in member_names and 'Technical_Sales' in member_names:
            collaboration_bonus += 0.15
        
        # Technical Sales + Sales Engineer: Technical depth
        if 'Technical_Sales' in member_names and 'Sales_Engineer' in member_names:
            collaboration_bonus += 0.10
        
        # Sales Director provides credibility and deal closure power
        if 'Sales_Director' in member_names and len(member_names) > 1:
            collaboration_bonus += 0.08
        
        # Full team bonus
        if len(member_names) == 4:
            collaboration_bonus += 0.05
        
        final_prob = min(1.0, base_prob + collaboration_bonus)
        
        # Convert probability to expected commission value
        return final_prob * total_commission
    
    # Calculate fair commission distribution
    players = list(sales_team.values())
    calculator = ShapleyValueCalculator(deal_contribution, players)
    shapley_values = calculator.calculate_shapley_values()
    
    print("Fair Commission Distribution:")
    team_members = list(sales_team.keys())
    total_distributed = sum(shapley_values.values())
    
    for i, (rate, commission) in enumerate(shapley_values.items()):
        member = team_members[i].replace('_', ' ')
        percentage = (commission / total_distributed) * 100
        
        print(f"  {member}:")
        print(f"    - Commission: ${commission:,.0f} ({percentage:.1f}%)")
        print(f"    - Multiple of base rate: {commission / (rate * total_commission):.1f}x")
    
    print(f"\nTotal distributed: ${total_distributed:,.0f}")
    
    # Show the value of collaboration
    individual_total = sum(rate * total_commission for rate in sales_team.values())
    collaboration_value = total_distributed - individual_total
    print(f"Value created through collaboration: ${collaboration_value:,.0f}")


def business_insights():
    """
    Summary of key business insights from Shapley value applications
    """
    print("\n" + "=" * 70)
    print("KEY BUSINESS INSIGHTS")
    print("=" * 70)
    
    insights = [
        "1. FAIRNESS: Shapley values ensure each participant receives their fair",
        "   share based on marginal contributions across all possible coalitions.",
        "",
        "2. EFFICIENCY: The total value distributed always equals the grand",
        "   coalition value, ensuring no value is lost in the allocation process.",
        "",
        "3. INCENTIVE ALIGNMENT: Players are rewarded for their actual",
        "   contributions, encouraging optimal effort and collaboration.",
        "",
        "4. NEGOTIATION TOOL: Provides objective basis for profit sharing,",
        "   cost allocation, and resource distribution negotiations.",
        "",
        "5. STRATEGIC INSIGHTS: Reveals which combinations create the most",
        "   value, informing partnership and team formation decisions.",
        "",
        "6. TRANSPARENCY: Mathematical foundation provides clear rationale",
        "   for allocation decisions, reducing disputes and improving trust."
    ]
    
    for insight in insights:
        print(insight)


if __name__ == "__main__":
    joint_venture_example()
    cost_allocation_example()
    sales_team_example()
    business_insights()